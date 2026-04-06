"""
Round 3: Few-shot + numeric confidence + feature-augmented approaches.
Goal: push correlation from 0.40 toward 0.71 to achieve Var(Y-Yhat)/Var(Y) ≈ 0.5.
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
RESULTS_PATH = "data/round3_results.json"
SAMPLE_SIZE = 300
SEED = 42

df_full = pd.read_csv(INPUT_PATH)
np.random.seed(SEED)
sample_idx = np.random.choice(len(df_full), size=SAMPLE_SIZE, replace=False)
df_sample = df_full.iloc[sample_idx].reset_index(drop=True)
Y = df_sample["vote"].values
VAR_Y = float(np.var(Y))
TARGET = VAR_Y / 2

# Get some labeled examples for few-shot (NOT from the test sample)
other_idx = np.setdiff1d(np.arange(len(df_full)), sample_idx)
np.random.shuffle(other_idx)
fewshot_idx = other_idx[:8]
df_fewshot = df_full.iloc[fewshot_idx].reset_index(drop=True)

print(f"Sample: {SAMPLE_SIZE} | Var(Y) = {VAR_Y:.6f} | Target = {TARGET:.6f}")
print()


def make_user_msg(question, ans_a, ans_b):
    a = ans_a if pd.notna(ans_a) else "(no answer provided)"
    b = ans_b if pd.notna(ans_b) else "(no answer provided)"
    return f"Question: {question}\n\nAnswer A: {a}\n\nAnswer B: {b}"


def parse_verdict(text):
    """Parse verdict from model output, return score in [0,1]."""
    text_upper = text.strip().upper()
    last_line = text_upper.split("\n")[-1].strip()

    # Check for numeric probability first
    for line in [last_line] + text_upper.split("\n")[-3:]:
        line = line.strip()
        m = re.search(r"(?:PROBABILITY|SCORE|CONFIDENCE|P\(B\)|P_B)[:\s=]*([01]\.?\d*)", line)
        if m:
            val = float(m.group(1))
            if 0 <= val <= 1:
                return val

    # Check for VERDICT with confidence
    for line in [last_line, text_upper]:
        m = re.search(r"VERDICT:\s*([AB])\s*\((\d+)%?\)", line)
        if m:
            letter, conf = m.group(1), int(m.group(2))
            conf_score = min(conf, 100) / 100.0
            if letter == "A":
                return (1 - conf_score) / 2  # high confidence A → near 0
            else:
                return 0.5 + conf_score / 2  # high confidence B → near 1

        m = re.search(r"VERDICT:\s*([AB])\s*(STRONG|WEAK)?", line)
        if m:
            letter, strength = m.group(1), m.group(2)
            if letter == "A":
                return 0.0 if strength == "STRONG" else 0.15 if strength == "WEAK" else 0.0
            else:
                return 1.0 if strength == "STRONG" else 0.85 if strength == "WEAK" else 1.0
        if "VERDICT:" in line and "TIE" in line:
            return 0.5

    # Standard A/B/TIE
    for line in [last_line, text_upper]:
        m = re.search(r"(?:BETTER|WINNER|VERDICT|ANSWER)[:\s]*([AB]|TIE)\b", line)
        if m:
            v = m.group(1)
            return 0.0 if v == "A" else 1.0 if v == "B" else 0.5
        if re.match(r"^[AB]$", line):
            return 0.0 if line == "A" else 1.0
        if "TIE" in line:
            return 0.5

    # Try to find any number on the last line
    m = re.search(r"(\d\.?\d*)", last_line)
    if m:
        val = float(m.group(1))
        if 0 <= val <= 1:
            return val

    if text_upper.startswith("A"):
        return 0.0
    if text_upper.startswith("B"):
        return 1.0
    return None


async def get_predictions_debiased(client, model, system_prompt, label="",
                                    fewshot_messages=None, max_tokens=512):
    """Get debiased predictions with optional few-shot."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    scores_fwd = [None] * SAMPLE_SIZE
    scores_rev = [None] * SAMPLE_SIZE

    async def process_one(i, row, is_forward):
        async with semaphore:
            if is_forward:
                msg = make_user_msg(row["question"], row["answer_0"], row["answer_1"])
            else:
                msg = make_user_msg(row["question"], row["answer_1"], row["answer_0"])

            messages = []
            if fewshot_messages:
                messages.extend(fewshot_messages)
            messages.append({"role": "user", "content": msg})

            for attempt in range(3):
                try:
                    response = await client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        system=system_prompt,
                        messages=messages,
                    )
                    raw = parse_verdict(response.content[0].text)
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

    n_valid = np.sum(~np.isnan(Yhat))
    print(f"  [{label}] {n_valid}/{SAMPLE_SIZE} valid")
    return Yhat


def evaluate(Y, Yhat, label=""):
    mask = ~np.isnan(Yhat)
    Y_s, Yh_s = Y[mask], Yhat[mask]
    var_y = np.var(Y_s)
    var_resid = np.var(Y_s - Yh_s)
    corr = np.corrcoef(Y_s, Yh_s)[0, 1] if np.std(Yh_s) > 1e-10 else 0
    cov = np.cov(Y_s, Yh_s)[0, 1]
    var_yhat = np.var(Yh_s)
    alpha = cov / var_yhat if var_yhat > 1e-10 else 0
    Yhat_cal = np.mean(Y_s) + alpha * (Yh_s - np.mean(Yh_s))
    var_resid_cal = np.var(Y_s - Yhat_cal)
    raw_ratio = var_resid / var_y if var_y > 0 else 999
    cal_ratio = var_resid_cal / var_y if var_y > 0 else 999
    print(f"  [{label}] raw_ratio={raw_ratio:.3f} cal_ratio={cal_ratio:.3f} corr={corr:.3f} alpha={alpha:.3f} Yhat μ={np.mean(Yh_s):.3f} σ={np.std(Yh_s):.3f}")
    return {
        "raw_var_resid": float(var_resid), "raw_ratio": float(raw_ratio),
        "cal_var_resid": float(var_resid_cal), "cal_ratio": float(cal_ratio),
        "corr": float(corr), "alpha": float(alpha), "n_valid": int(mask.sum()),
        "yhat_mean": float(np.mean(Yh_s)), "yhat_std": float(np.std(Yh_s)),
    }


# ============================================================
# Build few-shot examples
# ============================================================
def build_fewshot_messages(n_examples=6):
    """Build few-shot conversation from labeled examples."""
    messages = []
    for i in range(min(n_examples, len(df_fewshot))):
        row = df_fewshot.iloc[i]
        vote = row["vote"]
        q = row["question"]
        a0 = row["answer_0"] if pd.notna(row["answer_0"]) else "(no answer provided)"
        a1 = row["answer_1"] if pd.notna(row["answer_1"]) else "(no answer provided)"

        user_msg = f"Question: {q}\n\nAnswer A: {a0}\n\nAnswer B: {a1}"

        if vote < 0.35:
            response = "VERDICT: A"
        elif vote > 0.65:
            response = "VERDICT: B"
        else:
            response = "VERDICT: TIE"

        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": response})
    return messages


def build_fewshot_graded(n_examples=6):
    """Few-shot with graded verdicts."""
    messages = []
    for i in range(min(n_examples, len(df_fewshot))):
        row = df_fewshot.iloc[i]
        vote = row["vote"]
        q = row["question"]
        a0 = row["answer_0"] if pd.notna(row["answer_0"]) else "(no answer provided)"
        a1 = row["answer_1"] if pd.notna(row["answer_1"]) else "(no answer provided)"

        user_msg = f"Question: {q}\n\nAnswer A: {a0}\n\nAnswer B: {a1}"

        if vote < 0.2:
            response = "VERDICT: A STRONG"
        elif vote < 0.4:
            response = "VERDICT: A WEAK"
        elif vote < 0.6:
            response = "VERDICT: TIE"
        elif vote < 0.8:
            response = "VERDICT: B WEAK"
        else:
            response = "VERDICT: B STRONG"

        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": response})
    return messages


# ============================================================
# Prompts
# ============================================================

EXPERIMENTS = {}

EXPERIMENTS["fewshot_binary_haiku"] = {
    "model": HAIKU_MODEL,
    "system": """\
Which answer is better? Consider accuracy, relevance, and evidence quality. Empty answers always lose.

Reply with ONLY:
VERDICT: A
VERDICT: B
VERDICT: TIE""",
    "fewshot_fn": lambda: build_fewshot_messages(6),
}

EXPERIMENTS["fewshot_graded_haiku"] = {
    "model": HAIKU_MODEL,
    "system": """\
Compare the two answers. Consider accuracy, relevance, evidence, and completeness.
Empty or missing answers always lose.

Reply with ONLY one of:
VERDICT: A STRONG
VERDICT: A WEAK
VERDICT: TIE
VERDICT: B WEAK
VERDICT: B STRONG""",
    "fewshot_fn": lambda: build_fewshot_graded(6),
}

EXPERIMENTS["fewshot_graded_sonnet"] = {
    "model": SONNET_MODEL,
    "system": """\
Compare the two answers. Consider accuracy, relevance, evidence, and completeness.
Empty or missing answers always lose.

Reply with ONLY one of:
VERDICT: A STRONG
VERDICT: A WEAK
VERDICT: TIE
VERDICT: B WEAK
VERDICT: B STRONG""",
    "fewshot_fn": lambda: build_fewshot_graded(6),
}

EXPERIMENTS["probability_haiku"] = {
    "model": HAIKU_MODEL,
    "system": """\
You are predicting human preferences in a QA evaluation dataset.

Given a question and two answers (A and B), estimate the probability that human raters preferred Answer B.

Consider: factual accuracy (most important), relevance, evidence/citations, completeness.
An empty answer always loses.

Output ONLY a single number between 0.0 and 1.0 on one line.
0.0 = humans definitely preferred A
0.5 = equal preference
1.0 = humans definitely preferred B""",
    "fewshot_fn": None,
}

EXPERIMENTS["probability_sonnet"] = {
    "model": SONNET_MODEL,
    "system": """\
You are predicting human preferences in a QA evaluation dataset.

Given a question and two answers (A and B), estimate the probability that human raters preferred Answer B.

Consider: factual accuracy (most important), relevance, evidence/citations, completeness.
An empty answer always loses.

Output ONLY a single number between 0.0 and 1.0 on one line.
0.0 = humans definitely preferred A
0.5 = equal preference
1.0 = humans definitely preferred B""",
    "fewshot_fn": None,
}

EXPERIMENTS["fewshot_probability_sonnet"] = {
    "model": SONNET_MODEL,
    "system": """\
You are predicting human preferences in a QA evaluation dataset. Given a question and two answers, estimate the probability that human raters preferred Answer B over Answer A.

Consider: factual accuracy (most important), relevance, evidence/citations, completeness. An empty answer always loses.

Output ONLY a number between 0.0 and 1.0.""",
    "fewshot_fn": lambda: _build_prob_fewshot(6),
}

EXPERIMENTS["cot_probability_sonnet"] = {
    "model": SONNET_MODEL,
    "system": """\
You are predicting human preferences for a QA evaluation dataset.

Analyze both answers carefully:
1. Check for factual errors or wrong information
2. Check if each answer addresses the specific question
3. Check citation/reference quality
4. Consider completeness and clarity

Then estimate the probability that human raters preferred Answer B.

After your analysis, write on the LAST line:
SCORE: X.XX
where X.XX is between 0.00 (humans preferred A) and 1.00 (humans preferred B).""",
    "fewshot_fn": None,
    "max_tokens": 1024,
}

EXPERIMENTS["cot_5level_sonnet"] = {
    "model": SONNET_MODEL,
    "system": """\
You are predicting which answer human evaluators preferred in a QA dataset.

Analyze both answers:
1. Factual correctness (most important to human raters)
2. Does it answer the specific question asked?
3. Are claims supported by cited references?
4. Is it complete without excessive padding?
5. Empty/missing answers always lose.

After your analysis, give your confidence-weighted verdict on the LAST line.
Use EXACTLY one of these formats:
VERDICT: A STRONG
VERDICT: A WEAK
VERDICT: TIE
VERDICT: B WEAK
VERDICT: B STRONG""",
    "fewshot_fn": None,
    "max_tokens": 1024,
}


def _build_prob_fewshot(n):
    messages = []
    for i in range(min(n, len(df_fewshot))):
        row = df_fewshot.iloc[i]
        vote = row["vote"]
        q = row["question"]
        a0 = row["answer_0"] if pd.notna(row["answer_0"]) else "(no answer provided)"
        a1 = row["answer_1"] if pd.notna(row["answer_1"]) else "(no answer provided)"
        user_msg = f"Question: {q}\n\nAnswer A: {a0}\n\nAnswer B: {a1}"
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": f"{vote:.2f}"})
    return messages


# ============================================================
# Also try pure feature-based baseline
# ============================================================
def feature_baseline():
    """Simple heuristic: prefer longer answers with more citations."""
    Yhat = np.full(SAMPLE_SIZE, np.nan)
    for i, row in df_sample.iterrows():
        a0 = str(row["answer_0"]) if pd.notna(row["answer_0"]) else ""
        a1 = str(row["answer_1"]) if pd.notna(row["answer_1"]) else ""

        len0, len1 = len(a0), len(a1)
        cit0 = len(re.findall(r'\[\d+\]', a0))
        cit1 = len(re.findall(r'\[\d+\]', a1))

        # Simple logistic-style score
        len_diff = (len1 - len0) / max(len0 + len1, 1)
        cit_diff = (cit1 - cit0) / max(cit0 + cit1, 1)

        # Empty check
        if len0 == 0 and len1 > 0:
            Yhat[i] = 0.9
        elif len1 == 0 and len0 > 0:
            Yhat[i] = 0.1
        else:
            # Weighted combination
            score = 0.5 + 0.3 * len_diff + 0.2 * cit_diff
            Yhat[i] = np.clip(score, 0, 1)
    return Yhat


async def main():
    client = anthropic.AsyncAnthropic()

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} prior results\n")
    else:
        all_results = {}

    # Feature baseline
    if "feature_baseline" not in all_results:
        print("Testing: feature_baseline")
        Yhat = feature_baseline()
        metrics = evaluate(Y, Yhat, label="feature_baseline")
        all_results["feature_baseline"] = metrics
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

    # LLM experiments
    for name, config in EXPERIMENTS.items():
        if name in all_results and all_results[name].get("corr") is not None:
            r = all_results[name]
            print(f"[{name}] (cached) cal_ratio={r['cal_ratio']:.3f} corr={r['corr']:.3f}")
            continue

        print(f"\nTesting: {name} ({config['model'].split('-')[1]})")
        t0 = time.time()
        fewshot = config.get("fewshot_fn")
        fewshot_msgs = fewshot() if fewshot else None
        max_tokens = config.get("max_tokens", 512)

        try:
            preds = await get_predictions_debiased(
                client, config["model"], config["system"],
                label=name, fewshot_messages=fewshot_msgs, max_tokens=max_tokens,
            )
            metrics = evaluate(Y, preds, label=name)
            metrics["predictions"] = [float(v) if not np.isnan(v) else None for v in preds]
            metrics["elapsed_s"] = time.time() - t0
            all_results[name] = metrics
        except Exception as e:
            print(f"  [{name}] Error: {e}")
            all_results[name] = {"corr": None, "cal_ratio": None, "error": str(e)}

        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  (checkpoint saved, {time.time()-t0:.0f}s)")

    # ============================================================
    # Ensemble: combine best LLM predictions with features
    # ============================================================
    print(f"\n{'='*70}")
    print("ENSEMBLES")
    print(f"{'='*70}")

    # Collect all valid prediction arrays
    pred_arrays = {}
    for name, r in all_results.items():
        if "predictions" in r:
            arr = np.array([v if v is not None else np.nan for v in r["predictions"]])
            if np.sum(~np.isnan(arr)) > SAMPLE_SIZE * 0.8:
                pred_arrays[name] = arr

    if len(pred_arrays) >= 2:
        # All LLM ensemble
        stacked = np.stack(list(pred_arrays.values()))
        ens = np.nanmean(stacked, axis=0)
        print(f"\n  Ensemble all ({len(pred_arrays)} models):")
        metrics = evaluate(Y, ens, label="ensemble_all")
        all_results["ensemble_all_round3"] = metrics

        # Top-3 by calibrated ratio
        sorted_names = sorted(pred_arrays.keys(),
            key=lambda k: all_results[k].get("cal_ratio", 999))
        top3 = sorted_names[:3]
        print(f"\n  Ensemble top-3: {top3}")
        ens3 = np.nanmean(np.stack([pred_arrays[k] for k in top3]), axis=0)
        metrics = evaluate(Y, ens3, label="ensemble_top3")
        all_results["ensemble_top3_round3"] = metrics

        # Top by correlation
        sorted_by_corr = sorted(pred_arrays.keys(),
            key=lambda k: all_results[k].get("corr", 0), reverse=True)
        top3_corr = sorted_by_corr[:3]
        print(f"\n  Ensemble top-3 by corr: {top3_corr}")
        ens3c = np.nanmean(np.stack([pred_arrays[k] for k in top3_corr]), axis=0)
        metrics = evaluate(Y, ens3c, label="ensemble_top3_corr")
        all_results["ensemble_top3_corr_round3"] = metrics

        # Feature + top LLM
        if "feature_baseline" in all_results:
            feat_preds = feature_baseline()
            best_llm = sorted_by_corr[0]
            combined = np.nanmean(np.stack([feat_preds, pred_arrays[best_llm]]), axis=0)
            print(f"\n  Feature + {best_llm}:")
            metrics = evaluate(Y, combined, label="feature_plus_llm")
            all_results["feature_plus_best_llm"] = metrics

        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

    # ============================================================
    # Final ranking
    # ============================================================
    print(f"\n{'='*70}")
    print(f"FINAL RANKING (target cal_ratio = 0.500)")
    print(f"{'='*70}")
    valid = {k: v for k, v in all_results.items()
             if v.get("cal_ratio") is not None and v.get("corr") is not None}
    ranking = sorted(valid, key=lambda k: valid[k]["cal_ratio"])
    print(f"{'Name':45s} {'raw':>8s} {'cal':>8s} {'corr':>8s}")
    print("-" * 72)
    for name in ranking:
        r = valid[name]
        print(f"  {name:43s} {r['raw_ratio']:8.3f} {r['cal_ratio']:8.3f} {r['corr']:8.3f}")

    best = ranking[0]
    print(f"\nBest: {best}")
    print(f"  Calibrated ratio = {valid[best]['cal_ratio']:.3f}")
    print(f"  Correlation      = {valid[best]['corr']:.3f}")
    print(f"  Target ratio     = 0.500")


if __name__ == "__main__":
    asyncio.run(main())
