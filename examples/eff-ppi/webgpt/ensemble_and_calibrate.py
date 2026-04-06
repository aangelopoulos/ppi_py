"""
Round 3: Ensemble top prompts + optimal calibration + Sonnet evaluation.

Strategy:
1. Run the top-5 highest-correlation prompts, collect per-example predictions
2. Ensemble them (average) to boost effective correlation
3. Apply optimal linear shrinkage: Yhat_cal = mean(Y) + alpha*(Yhat - mean(Yhat))
   where alpha = Cov(Y,Yhat)/Var(Yhat) minimizes Var(Y - Yhat_cal)
4. Test on Sonnet to see if a better model helps more
5. Save the best approach for the full run
"""

import asyncio
import json
import os
import re
import time

import anthropic
import numpy as np
import pandas as pd

HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-4-sonnet-20250514"
MAX_CONCURRENT = 50
INPUT_PATH = "data/webgpt_comparisons.csv"
ENSEMBLE_PATH = "data/ensemble_results.json"
SAMPLE_SIZE = 300
SEED = 42

# ============================================================
# Data
# ============================================================
df_full = pd.read_csv(INPUT_PATH)
np.random.seed(SEED)
sample_idx = np.random.choice(len(df_full), size=SAMPLE_SIZE, replace=False)
df_sample = df_full.iloc[sample_idx].reset_index(drop=True)
Y = df_sample["vote"].values
VAR_Y = float(np.var(Y))
TARGET = VAR_Y / 2

print(f"Sample: {SAMPLE_SIZE} | Var(Y) = {VAR_Y:.6f} | Target = {TARGET:.6f}")
print()


def make_user_msg(question, ans_first, ans_second):
    a0 = ans_first if pd.notna(ans_first) else "(no answer provided)"
    a1 = ans_second if pd.notna(ans_second) else "(no answer provided)"
    return f"Question: {question}\n\nAnswer A: {a0}\n\nAnswer B: {a1}"


def parse_ab_verdict(text):
    """Parse A/B/TIE from model output, return score in [0,1]."""
    text_upper = text.strip().upper()
    last_line = text_upper.split("\n")[-1].strip()

    for line in [last_line, text_upper]:
        m = re.search(r"VERDICT:\s*([AB])\s*(STRONG|WEAK)?", line)
        if m:
            letter, strength = m.group(1), m.group(2)
            if letter == "A":
                return 0.0 if strength == "STRONG" else 0.25 if strength == "WEAK" else 0.0
            else:
                return 1.0 if strength == "STRONG" else 0.75 if strength == "WEAK" else 1.0
        if "VERDICT:" in line and "TIE" in line:
            return 0.5

    for line in [last_line, text_upper]:
        m = re.search(r"(?:BETTER|WINNER|VERDICT|ANSWER)[:\s]*([AB]|TIE)\b", line)
        if m:
            v = m.group(1)
            return 0.0 if v == "A" else 1.0 if v == "B" else 0.5
        if re.match(r"^[AB]$", line):
            return 0.0 if line == "A" else 1.0
        if "TIE" in line:
            return 0.5

    if text_upper.startswith("A"):
        return 0.0
    if text_upper.startswith("B"):
        return 1.0
    return None


async def get_predictions(client, model, system_prompt, label=""):
    """Get debiased predictions (forward + reverse) for the sample."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    scores_fwd = [None] * SAMPLE_SIZE
    scores_rev = [None] * SAMPLE_SIZE

    async def process_one(i, row, is_forward):
        async with semaphore:
            if is_forward:
                msg = make_user_msg(row["question"], row["answer_0"], row["answer_1"])
            else:
                msg = make_user_msg(row["question"], row["answer_1"], row["answer_0"])

            for attempt in range(3):
                try:
                    response = await client.messages.create(
                        model=model,
                        max_tokens=512,
                        system=system_prompt,
                        messages=[{"role": "user", "content": msg}],
                    )
                    raw = parse_ab_verdict(response.content[0].text)
                    if raw is None:
                        if attempt < 2:
                            continue
                        return
                    if is_forward:
                        scores_fwd[i] = raw
                    else:
                        scores_rev[i] = 1.0 - raw
                    return
                except anthropic.RateLimitError:
                    await asyncio.sleep(2 ** (attempt + 1))
                except anthropic.APIStatusError as e:
                    if "credit balance" in str(e):
                        raise
                    await asyncio.sleep(2 ** (attempt + 1))
                except Exception:
                    if attempt < 2:
                        continue
                    return

    tasks = []
    for i, row in df_sample.iterrows():
        tasks.append(process_one(i, row, True))
        tasks.append(process_one(i, row, False))

    await asyncio.gather(*tasks)

    Yhat = np.full(SAMPLE_SIZE, np.nan)
    for i in range(SAMPLE_SIZE):
        vals = [v for v in [scores_fwd[i], scores_rev[i]] if v is not None]
        if vals:
            Yhat[i] = np.mean(vals)

    mask = ~np.isnan(Yhat)
    print(f"  [{label}] {mask.sum()}/{SAMPLE_SIZE} valid")
    return Yhat


def evaluate(Y, Yhat, label=""):
    """Compute metrics + optimal calibration."""
    mask = ~np.isnan(Yhat)
    Y_s, Yh_s = Y[mask], Yhat[mask]

    # Raw metrics
    var_resid = np.var(Y_s - Yh_s)
    corr = np.corrcoef(Y_s, Yh_s)[0, 1] if np.std(Yh_s) > 0 else 0
    raw_ratio = var_resid / np.var(Y_s)

    # Optimal linear calibration: Yhat_cal = mean(Y) + alpha*(Yhat - mean(Yhat))
    # alpha* = Cov(Y,Yhat) / Var(Yhat)
    cov = np.cov(Y_s, Yh_s)[0, 1]
    var_yhat = np.var(Yh_s)
    alpha = cov / var_yhat if var_yhat > 0 else 0
    Yhat_cal = np.mean(Y_s) + alpha * (Yh_s - np.mean(Yh_s))
    var_resid_cal = np.var(Y_s - Yhat_cal)
    cal_ratio = var_resid_cal / np.var(Y_s)
    # Note: var_resid_cal = Var(Y) * (1 - corr^2) theoretically

    print(f"  [{label}] raw: Var(Y-Yhat)={var_resid:.6f} ratio={raw_ratio:.3f} | "
          f"calibrated: Var(Y-Yhat)={var_resid_cal:.6f} ratio={cal_ratio:.3f} | "
          f"corr={corr:.3f} alpha={alpha:.3f}")

    return {
        "raw_var_resid": float(var_resid), "raw_ratio": float(raw_ratio),
        "cal_var_resid": float(var_resid_cal), "cal_ratio": float(cal_ratio),
        "corr": float(corr), "alpha": float(alpha),
        "n_valid": int(mask.sum()),
    }


# Top prompts by correlation from round 2
TOP_PROMPTS = {
    "v16_tiebreak_bias": """\
Which answer is better? Most answer pairs have a clear winner — ties are rare.

Consider accuracy, relevance, and evidence. Empty = automatic loss.
Pick one:
VERDICT: A
VERDICT: B""",

    "v15_pairwise_rubric": """\
Score each answer independently on a 1-5 scale, then compare.

For each answer, rate:
- Correctness (1-5): Are the facts right?
- Relevance (1-5): Does it answer the question?
- Evidence (1-5): Are claims cited?

An empty answer scores 0 on all criteria.

After scoring both answers, pick the one with the higher total score.
Last line:
VERDICT: A
VERDICT: B
VERDICT: TIE""",

    "v10_cot_accuracy_first": """\
Compare these two answers. The MOST important criterion is factual accuracy.

Think through:
1. Is either answer empty or missing? (automatic loss)
2. Does either answer contain factually wrong information?
3. Does each answer actually address the specific question?
4. Which is better supported by references?

Then give your verdict on the LAST line:
VERDICT: A
VERDICT: B
VERDICT: TIE""",

    "v13_webgpt_specific": """\
You are evaluating answers from WebGPT, a QA system that browses the web to answer questions.

These answers often include references like [1], [2] from web sources. Human raters evaluated pairs of these answers.

Human raters cared about:
1. Did the system find the correct answer? (most important)
2. Are the claims backed by the cited sources?
3. Is the answer relevant and complete without being padded?
4. An empty or missing answer is always rated worse.

Predict which answer the human rater preferred.
Last line:
VERDICT: A
VERDICT: B
VERDICT: TIE""",

    "v8_calibrated": """\
You are calibrating human preference predictions for a dataset of question-answer comparisons.

In this dataset, Answer A is preferred about 50% of the time and Answer B about 50% of the time. Your job is to predict which one humans actually chose as better.

Humans judge based on: accuracy, relevance, helpfulness, and citation quality. An empty answer always loses.

Think briefly, then give your prediction on the last line:
VERDICT: A
VERDICT: B
VERDICT: TIE""",
}


async def main():
    client = anthropic.AsyncAnthropic()

    # Load prior results
    if os.path.exists(ENSEMBLE_PATH):
        with open(ENSEMBLE_PATH) as f:
            all_results = json.load(f)
        print(f"Loaded prior results from {ENSEMBLE_PATH}\n")
    else:
        all_results = {}

    # ============================================================
    # Phase 1: Collect predictions from top Haiku prompts
    # ============================================================
    print("=" * 70)
    print("PHASE 1: Collecting predictions from top Haiku prompts")
    print("=" * 70)

    haiku_preds = {}  # name -> np.array of predictions
    for name, prompt in TOP_PROMPTS.items():
        cache_key = f"haiku_{name}"
        if cache_key in all_results and "predictions" in all_results[cache_key]:
            preds = np.array(all_results[cache_key]["predictions"])
            print(f"  [{name}] loaded from cache ({np.sum(~np.isnan(preds))}/{SAMPLE_SIZE} valid)")
            haiku_preds[name] = preds
            continue

        print(f"\n  Collecting: {name}")
        t0 = time.time()
        try:
            preds = await get_predictions(client, HAIKU_MODEL, prompt, label=name)
            haiku_preds[name] = preds
            metrics = evaluate(Y, preds, label=name)
            all_results[cache_key] = {
                "predictions": [float(v) if not np.isnan(v) else None for v in preds],
                **metrics,
                "elapsed_s": time.time() - t0,
            }
        except Exception as e:
            print(f"  [{name}] Error: {e}")
            continue

        with open(ENSEMBLE_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  (checkpoint saved)")

    # ============================================================
    # Phase 2: Ensemble
    # ============================================================
    if len(haiku_preds) >= 2:
        print(f"\n{'='*70}")
        print(f"PHASE 2: Ensembles")
        print(f"{'='*70}")

        # All-ensemble
        all_pred_arrays = list(haiku_preds.values())
        stacked = np.stack(all_pred_arrays)
        ensemble_all = np.nanmean(stacked, axis=0)
        metrics = evaluate(Y, ensemble_all, label="ensemble_all_haiku")
        all_results["ensemble_all_haiku"] = metrics

        # Best-3 ensemble (by correlation)
        sorted_by_corr = sorted(haiku_preds.keys(),
            key=lambda k: all_results.get(f"haiku_{k}", {}).get("corr", 0), reverse=True)
        top3 = sorted_by_corr[:3]
        print(f"  Top-3 by corr: {top3}")
        stacked3 = np.stack([haiku_preds[k] for k in top3])
        ensemble_top3 = np.nanmean(stacked3, axis=0)
        metrics = evaluate(Y, ensemble_top3, label="ensemble_top3_haiku")
        all_results["ensemble_top3_haiku"] = metrics

        with open(ENSEMBLE_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

    # ============================================================
    # Phase 3: Test with Sonnet (higher capability model)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"PHASE 3: Testing Sonnet (higher capability)")
    print(f"{'='*70}")

    # Use the best prompt from round 2
    sonnet_prompts = {
        "sonnet_v13_webgpt": TOP_PROMPTS["v13_webgpt_specific"],
        "sonnet_v10_accuracy": TOP_PROMPTS["v10_cot_accuracy_first"],
        "sonnet_v15_rubric": TOP_PROMPTS["v15_pairwise_rubric"],
    }

    for name, prompt in sonnet_prompts.items():
        if name in all_results and "predictions" in all_results[name]:
            preds = np.array([v if v is not None else np.nan for v in all_results[name]["predictions"]])
            n_valid = np.sum(~np.isnan(preds))
            print(f"  [{name}] loaded from cache ({n_valid}/{SAMPLE_SIZE} valid)")
            continue

        print(f"\n  Collecting: {name}")
        t0 = time.time()
        try:
            preds = await get_predictions(client, SONNET_MODEL, prompt, label=name)
            metrics = evaluate(Y, preds, label=name)
            all_results[name] = {
                "predictions": [float(v) if not np.isnan(v) else None for v in preds],
                **metrics,
                "elapsed_s": time.time() - t0,
            }
        except Exception as e:
            print(f"  [{name}] Error: {e}")
            continue

        with open(ENSEMBLE_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  (checkpoint saved)")

    # ============================================================
    # Final summary
    # ============================================================
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY (target ratio = 0.500)")
    print(f"{'='*70}")
    print(f"{'Name':40s} {'raw_ratio':>10s} {'cal_ratio':>10s} {'corr':>8s}")
    print("-" * 70)
    for name in sorted(all_results, key=lambda k: all_results[k].get("cal_ratio", 999)):
        r = all_results[name]
        if "cal_ratio" not in r:
            continue
        print(f"  {name:38s} {r['raw_ratio']:10.3f} {r['cal_ratio']:10.3f} {r['corr']:8.3f}")

    # Find overall best
    best = min(
        (k for k in all_results if "cal_ratio" in all_results[k]),
        key=lambda k: all_results[k]["cal_ratio"],
    )
    print(f"\nBest: {best}")
    print(f"  Raw Var(Y-Yhat)        = {all_results[best].get('raw_var_resid', 'N/A')}")
    print(f"  Calibrated Var(Y-Yhat) = {all_results[best]['cal_var_resid']:.6f}")
    print(f"  Calibrated ratio       = {all_results[best]['cal_ratio']:.3f}")
    print(f"  Correlation            = {all_results[best]['corr']:.3f}")

    with open(ENSEMBLE_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {ENSEMBLE_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
