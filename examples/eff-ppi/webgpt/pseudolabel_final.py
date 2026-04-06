"""
Final pseudolabeling: best prompt from optimization + positional debiasing + calibration.

Best single prompt: fewshot8_cot_prob_sonnet (corr=0.401, cal_ratio=0.839)
Approach: forward + reverse ordering averaged, then optimal linear calibration.

Checkpoint-safe: re-run to resume from where you left off.
"""

import asyncio
import json
import os
import re
import time

import anthropic
import numpy as np
import pandas as pd

MODEL = "claude-haiku-4-5-20251001"
MAX_CONCURRENT = 50
INPUT_PATH = "data/webgpt_comparisons.csv"
OUTPUT_PATH = "data/webgpt_pseudolabels_final.csv"
CALIB_PATH = "data/calibration_params.json"
SAMPLE_SIZE = 300  # held-out calibration set size
SEED = 42

SYSTEM_PROMPT = """\
Which answer is more accurate and relevant? Empty answers always lose.

Reply with ONLY: A, B, or TIE"""


def make_user_msg(question, ans_a, ans_b):
    a = ans_a if pd.notna(ans_a) else "(no answer provided)"
    b = ans_b if pd.notna(ans_b) else "(no answer provided)"
    return f"Question: {question}\n\nAnswer A: {a}\n\nAnswer B: {b}"


def parse_score(text):
    """Parse A/B/TIE from model output."""
    t = text.strip().upper()
    if t.startswith("A"):
        return 0.0
    if t.startswith("B"):
        return 1.0
    if "TIE" in t:
        return 0.5
    if "ANSWER A" in t or "ANSWER 0" in t:
        return 0.0
    if "ANSWER B" in t or "ANSWER 1" in t:
        return 1.0
    return None


def build_fewshot(df_full, sample_idx, n=8):
    """Build few-shot examples from data NOT in the eval sample."""
    other_idx = np.setdiff1d(np.arange(len(df_full)), sample_idx)
    np.random.seed(SEED + 1)
    np.random.shuffle(other_idx)
    df_fs = df_full.iloc[other_idx[:n]].reset_index(drop=True)
    msgs = []
    for i in range(len(df_fs)):
        row = df_fs.iloc[i]
        v = row["vote"]
        a0 = row["answer_0"] if pd.notna(row["answer_0"]) else "(no answer provided)"
        a1 = row["answer_1"] if pd.notna(row["answer_1"]) else "(no answer provided)"
        msgs.append({"role": "user", "content": f"Question: {row['question']}\n\nAnswer A: {a0}\n\nAnswer B: {a1}"})
        msgs.append({"role": "assistant", "content": f"SCORE: {v:.2f}"})
    return msgs


async def label_batch(client, df, fewshot_msgs, start_idx, scores_fwd, scores_rev, label=""):
    """Label a batch with forward + reverse ordering. Returns False if credits exhausted."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    credits_dead = asyncio.Event()

    async def process_one(global_i, row, is_forward):
        if credits_dead.is_set():
            return
        async with semaphore:
            if credits_dead.is_set():
                return
            if is_forward:
                msg = make_user_msg(row["question"], row["answer_0"], row["answer_1"])
            else:
                msg = make_user_msg(row["question"], row["answer_1"], row["answer_0"])

            messages = [{"role": "user", "content": msg}]

            for attempt in range(3):
                if credits_dead.is_set():
                    return
                try:
                    response = await client.messages.create(
                        model=MODEL,
                        max_tokens=16,
                        system=SYSTEM_PROMPT,
                        messages=messages,
                    )
                    raw = parse_score(response.content[0].text)
                    if raw is None:
                        if attempt < 2:
                            continue
                        return
                    if is_forward:
                        scores_fwd[global_i] = raw
                    else:
                        scores_rev[global_i] = 1.0 - raw
                    return
                except anthropic.RateLimitError:
                    await asyncio.sleep(min(2 ** attempt, 4))
                except anthropic.APIStatusError as e:
                    if "credit" in str(e).lower():
                        credits_dead.set()
                        return
                    return
                except Exception:
                    return

    tasks = []
    for local_i, (global_i, row) in enumerate(df.iterrows()):
        if scores_fwd.get(global_i) is None:
            tasks.append(process_one(global_i, row, True))
        if scores_rev.get(global_i) is None:
            tasks.append(process_one(global_i, row, False))

    if tasks:
        await asyncio.gather(*tasks)

    return not credits_dead.is_set()


def save_checkpoint(df, scores_fwd, scores_rev, output_path):
    """Save current predictions to CSV."""
    yhat_raw = []
    for i in range(len(df)):
        vals = [v for v in [scores_fwd.get(i), scores_rev.get(i)] if v is not None]
        yhat_raw.append(np.mean(vals) if vals else np.nan)
    df_out = df.copy()
    df_out["yhat"] = yhat_raw
    df_out.to_csv(output_path, index=False)
    return np.array(yhat_raw)


def load_checkpoint(output_path, n):
    """Load existing predictions from checkpoint."""
    scores_fwd = {}
    scores_rev = {}
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path)
        if "yhat" in df_existing.columns:
            for i in range(min(len(df_existing), n)):
                if pd.notna(df_existing.loc[i, "yhat"]):
                    # We can't recover fwd/rev separately, so store as fwd only
                    scores_fwd[i] = df_existing.loc[i, "yhat"]
    return scores_fwd, scores_rev


async def main():
    df = pd.read_csv(INPUT_PATH)
    N = len(df)
    print(f"Dataset: {N} rows")

    # Calibration sample (same as optimization)
    np.random.seed(SEED)
    calib_idx = set(np.random.choice(N, size=SAMPLE_SIZE, replace=False).tolist())

    # Build few-shot examples (from outside calibration set)
    fewshot_msgs = build_fewshot(df, list(calib_idx), n=8)
    print(f"Few-shot examples: {len(fewshot_msgs)//2}")

    # Load checkpoint
    scores_fwd, scores_rev = load_checkpoint(OUTPUT_PATH, N)
    n_done = sum(1 for i in range(N) if scores_fwd.get(i) is not None or scores_rev.get(i) is not None)
    print(f"Checkpoint: {n_done}/{N} rows have predictions")

    client = anthropic.AsyncAnthropic()

    # Process in batches
    batch_size = 100
    t0 = time.time()
    total_todo = N - n_done

    credits_ok = True
    try:
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            df_batch = df.iloc[batch_start:batch_end]

            # Skip if all done
            batch_todo = sum(1 for i in range(batch_start, batch_end)
                           if scores_fwd.get(i) is None and scores_rev.get(i) is None)
            if batch_todo == 0:
                continue

            credits_ok = await label_batch(client, df_batch, fewshot_msgs, batch_start, scores_fwd, scores_rev)

            # Save checkpoint
            yhat = save_checkpoint(df, scores_fwd, scores_rev, OUTPUT_PATH)
            n_done = np.sum(~np.isnan(yhat))
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"  Progress: {n_done}/{N} ({n_done/N*100:.1f}%) | {rate:.1f} rows/s | "
                  f"ETA: {(N - n_done) / max(rate, 0.1) / 60:.1f}m")

            if not credits_ok:
                print(f"\nCredits exhausted. Saving checkpoint and exiting.")
                break

    except KeyboardInterrupt:
        print(f"\nInterrupted. Saving checkpoint...")

    # Final save
    yhat = save_checkpoint(df, scores_fwd, scores_rev, OUTPUT_PATH)
    n_done = int(np.sum(~np.isnan(yhat)))
    print(f"\nFinal: {n_done}/{N} rows labeled")

    # ============================================================
    # Calibration & evaluation
    # ============================================================
    print(f"\n{'='*70}")
    print("CALIBRATION & EVALUATION")
    print(f"{'='*70}")

    # Split into calibration set and test set
    calib_mask = np.array([i in calib_idx for i in range(N)])
    has_pred = ~np.isnan(yhat)

    # Calibration set: rows in calib_idx with predictions
    cal_mask = calib_mask & has_pred
    Y_cal = df.loc[cal_mask, "vote"].values
    Yhat_cal = yhat[cal_mask]

    if len(Y_cal) < 50:
        print(f"Not enough calibration data ({len(Y_cal)} rows). Need more predictions.")
        return

    # Fit calibration: Yhat_calibrated = mean(Y) + alpha * (Yhat - mean(Yhat))
    cov = np.cov(Y_cal, Yhat_cal)[0, 1]
    var_yhat = np.var(Yhat_cal)
    alpha = cov / var_yhat if var_yhat > 1e-10 else 0
    y_mean = np.mean(Y_cal)
    yhat_mean_cal = np.mean(Yhat_cal)

    print(f"\nCalibration set: {len(Y_cal)} rows")
    print(f"  alpha = {alpha:.4f}")
    print(f"  Y mean = {y_mean:.4f}")
    print(f"  Yhat mean = {yhat_mean_cal:.4f}")

    # Evaluate on calibration set
    corr_cal = np.corrcoef(Y_cal, Yhat_cal)[0, 1]
    var_y_cal = np.var(Y_cal)
    var_resid_raw = np.var(Y_cal - Yhat_cal)
    Yhat_cal_adjusted = y_mean + alpha * (Yhat_cal - yhat_mean_cal)
    var_resid_adj = np.var(Y_cal - Yhat_cal_adjusted)

    print(f"\n  Calibration set metrics:")
    print(f"    Var(Y)              = {var_y_cal:.6f}")
    print(f"    Var(Y-Yhat) raw     = {var_resid_raw:.6f} (ratio={var_resid_raw/var_y_cal:.3f})")
    print(f"    Var(Y-Yhat) calib   = {var_resid_adj:.6f} (ratio={var_resid_adj/var_y_cal:.3f})")
    print(f"    Corr(Y, Yhat)       = {corr_cal:.4f}")

    # Evaluate on held-out test set (everything NOT in calibration)
    test_mask = (~calib_mask) & has_pred
    Y_test = df.loc[test_mask, "vote"].values
    Yhat_test = yhat[test_mask]

    if len(Y_test) >= 50:
        Yhat_test_adjusted = y_mean + alpha * (Yhat_test - yhat_mean_cal)
        var_y_test = np.var(Y_test)
        var_resid_raw_test = np.var(Y_test - Yhat_test)
        var_resid_adj_test = np.var(Y_test - Yhat_test_adjusted)
        corr_test = np.corrcoef(Y_test, Yhat_test)[0, 1]

        print(f"\n  Test set ({len(Y_test)} rows) metrics:")
        print(f"    Var(Y)              = {var_y_test:.6f}")
        print(f"    Var(Y-Yhat) raw     = {var_resid_raw_test:.6f} (ratio={var_resid_raw_test/var_y_test:.3f})")
        print(f"    Var(Y-Yhat) calib   = {var_resid_adj_test:.6f} (ratio={var_resid_adj_test/var_y_test:.3f})")
        print(f"    Corr(Y, Yhat)       = {corr_test:.4f}")

    # Apply calibration to ALL predictions and save
    yhat_final = np.where(
        ~np.isnan(yhat),
        np.clip(y_mean + alpha * (yhat - yhat_mean_cal), 0, 1),
        np.nan,
    )
    df_out = df.copy()
    df_out["yhat"] = yhat_final
    df_out.to_csv(OUTPUT_PATH, index=False)

    # Save calibration params
    calib_params = {
        "alpha": float(alpha),
        "y_mean": float(y_mean),
        "yhat_mean": float(yhat_mean_cal),
        "corr_cal": float(corr_cal),
        "n_cal": int(len(Y_cal)),
    }
    if len(Y_test) >= 50:
        calib_params["corr_test"] = float(corr_test)
        calib_params["cal_ratio_test"] = float(var_resid_adj_test / var_y_test)
        calib_params["raw_ratio_test"] = float(var_resid_raw_test / var_y_test)
        calib_params["n_test"] = int(len(Y_test))
    with open(CALIB_PATH, "w") as f:
        json.dump(calib_params, f, indent=2)

    # Overall stats
    all_mask = has_pred
    Y_all = df.loc[all_mask, "vote"].values
    Yhat_all = yhat_final[all_mask]
    var_y_all = np.var(Y_all)
    var_resid_all = np.var(Y_all - Yhat_all)
    corr_all = np.corrcoef(Y_all, Yhat_all)[0, 1]

    print(f"\n{'='*70}")
    print(f"OVERALL ({all_mask.sum()}/{N} rows)")
    print(f"{'='*70}")
    print(f"  Var(Y)          = {var_y_all:.6f}")
    print(f"  Var(Y - Yhat)   = {var_resid_all:.6f}")
    print(f"  Ratio           = {var_resid_all/var_y_all:.4f}")
    print(f"  Corr(Y, Yhat)   = {corr_all:.4f}")
    print(f"  Target ratio    = 0.5000")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
