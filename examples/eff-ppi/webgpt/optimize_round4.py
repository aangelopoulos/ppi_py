"""
Round 4: Push correlation higher with aggressive strategies.

Key insight: corr=0.41 gives min cal_ratio=0.83. Need corr≈0.71 for ratio=0.50.

Strategies:
1. Extended CoT with explicit scoring rubric (forces careful evaluation)
2. Multi-aspect scoring (separate scores for accuracy, relevance, evidence → combine)
3. More few-shot examples (8 instead of 6)
4. Longer context window for reasoning
5. Two-pass: first analyze, then judge
"""

import asyncio
import json
import os
import re
import time

import anthropic
import numpy as np
import pandas as pd

SONNET_MODEL = "claude-4-sonnet-20250514"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
MAX_CONCURRENT = 30  # slightly lower to avoid rate limits
INPUT_PATH = "data/webgpt_comparisons.csv"
RESULTS_PATH = "data/round4_results.json"
SAMPLE_SIZE = 300
SEED = 42

df_full = pd.read_csv(INPUT_PATH)
np.random.seed(SEED)
sample_idx = np.random.choice(len(df_full), size=SAMPLE_SIZE, replace=False)
df_sample = df_full.iloc[sample_idx].reset_index(drop=True)
Y = df_sample["vote"].values
VAR_Y = float(np.var(Y))

other_idx = np.setdiff1d(np.arange(len(df_full)), sample_idx)
np.random.shuffle(other_idx)
df_fewshot = df_full.iloc[other_idx[:12]].reset_index(drop=True)

print(f"Sample: {SAMPLE_SIZE} | Var(Y) = {VAR_Y:.6f} | Target = {VAR_Y/2:.6f}")
print()


def make_user_msg(q, a, b):
    a = a if pd.notna(a) else "(no answer provided)"
    b = b if pd.notna(b) else "(no answer provided)"
    return f"Question: {q}\n\nAnswer A: {a}\n\nAnswer B: {b}"


def parse_score(text):
    """Parse numeric score from text. Returns float in [0,1] or None."""
    text_upper = text.strip().upper()
    lines = text_upper.split("\n")

    # Check last few lines for score patterns
    for line in reversed(lines[-5:]):
        line = line.strip()
        # SCORE: 0.35, P(B): 0.7, etc.
        m = re.search(r"(?:SCORE|P\(B\)|PROBABILITY|FINAL)[:\s=]*([01]\.?\d*)", line)
        if m:
            val = float(m.group(1))
            if 0 <= val <= 1:
                return val

        # VERDICT with confidence
        m = re.search(r"VERDICT:\s*([AB])\s*\((\d+)%\)", line)
        if m:
            letter, conf = m.group(1), int(m.group(2))
            if letter == "A":
                return max(0, 0.5 - conf / 200)
            else:
                return min(1, 0.5 + conf / 200)

        m = re.search(r"VERDICT:\s*([AB])\s*(STRONG|WEAK)?", line)
        if m:
            letter, strength = m.group(1), m.group(2)
            if letter == "A":
                return 0.0 if strength == "STRONG" else 0.15 if strength == "WEAK" else 0.0
            else:
                return 1.0 if strength == "STRONG" else 0.85 if strength == "WEAK" else 1.0
        if re.search(r"VERDICT.*TIE", line):
            return 0.5

    # Standard A/B/TIE
    last = lines[-1].strip()
    m = re.search(r"(?:BETTER|WINNER|VERDICT|ANSWER)[:\s]*([AB]|TIE)\b", last)
    if m:
        v = m.group(1)
        return 0.0 if v == "A" else 1.0 if v == "B" else 0.5
    if re.match(r"^[AB]$", last):
        return 0.0 if last == "A" else 1.0

    # Any number on last line
    m = re.search(r"(\d\.?\d*)", last)
    if m:
        val = float(m.group(1))
        if 0 <= val <= 1:
            return val

    return None


async def get_preds(client, model, system, label="", fewshot=None, max_tokens=1024):
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    fwd = [None] * SAMPLE_SIZE
    rev = [None] * SAMPLE_SIZE

    async def go(i, row, is_fwd):
        async with sem:
            if is_fwd:
                msg = make_user_msg(row["question"], row["answer_0"], row["answer_1"])
            else:
                msg = make_user_msg(row["question"], row["answer_1"], row["answer_0"])

            messages = list(fewshot) if fewshot else []
            messages.append({"role": "user", "content": msg})

            for attempt in range(3):
                try:
                    r = await client.messages.create(
                        model=model, max_tokens=max_tokens,
                        system=system, messages=messages,
                    )
                    raw = parse_score(r.content[0].text)
                    if raw is None:
                        if attempt < 2: continue
                        return
                    if is_fwd:
                        fwd[i] = raw
                    else:
                        rev[i] = 1.0 - raw
                    return
                except anthropic.RateLimitError:
                    await asyncio.sleep(2 ** (attempt + 1))
                except anthropic.APIStatusError as e:
                    if "credit balance" in str(e):
                        raise
                    await asyncio.sleep(2 ** (attempt + 1))
                except Exception:
                    if attempt < 2: continue
                    return

    tasks = []
    for i, row in df_sample.iterrows():
        tasks.append(go(i, row, True))
        tasks.append(go(i, row, False))
    await asyncio.gather(*tasks)

    Yhat = np.full(SAMPLE_SIZE, np.nan)
    for i in range(SAMPLE_SIZE):
        vals = [v for v in [fwd[i], rev[i]] if v is not None]
        if vals: Yhat[i] = np.mean(vals)
    n = np.sum(~np.isnan(Yhat))
    print(f"  [{label}] {n}/{SAMPLE_SIZE} valid")
    return Yhat


def evaluate(Y, Yhat, label=""):
    mask = ~np.isnan(Yhat)
    Y_s, Yh_s = Y[mask], Yhat[mask]
    var_y = np.var(Y_s)
    corr = np.corrcoef(Y_s, Yh_s)[0, 1] if np.std(Yh_s) > 1e-10 else 0
    cov = np.cov(Y_s, Yh_s)[0, 1]
    vYh = np.var(Yh_s)
    alpha = cov / vYh if vYh > 1e-10 else 0
    Yh_cal = np.mean(Y_s) + alpha * (Yh_s - np.mean(Yh_s))
    raw_r = np.var(Y_s - Yh_s) / var_y if var_y > 0 else 999
    cal_r = np.var(Y_s - Yh_cal) / var_y if var_y > 0 else 999
    print(f"  [{label}] raw={raw_r:.3f} cal={cal_r:.3f} corr={corr:.3f} α={alpha:.3f} μ={np.mean(Yh_s):.3f} σ={np.std(Yh_s):.3f}")
    return {
        "raw_ratio": float(raw_r), "cal_ratio": float(cal_r),
        "corr": float(corr), "alpha": float(alpha), "n_valid": int(mask.sum()),
        "raw_var_resid": float(np.var(Y_s - Yh_s)),
        "cal_var_resid": float(np.var(Y_s - Yh_cal)),
    }


def build_fewshot_prob(n=8):
    msgs = []
    for i in range(min(n, len(df_fewshot))):
        row = df_fewshot.iloc[i]
        v = row["vote"]
        q, a0, a1 = row["question"], row["answer_0"], row["answer_1"]
        a0 = a0 if pd.notna(a0) else "(no answer provided)"
        a1 = a1 if pd.notna(a1) else "(no answer provided)"
        msgs.append({"role": "user", "content": f"Question: {q}\n\nAnswer A: {a0}\n\nAnswer B: {a1}"})
        msgs.append({"role": "assistant", "content": f"SCORE: {v:.2f}"})
    return msgs


EXPERIMENTS = {}

EXPERIMENTS["multi_aspect_sonnet"] = {
    "model": SONNET_MODEL,
    "system": """\
Score each answer on three aspects (1-10 scale each), then compute an overall preference.

For EACH answer, evaluate:
ACCURACY (1-10): Are the stated facts correct? Any errors?
RELEVANCE (1-10): Does it answer the specific question asked?
EVIDENCE (1-10): Are claims supported by cited references [1], [2], etc.?

An empty/missing answer gets 0 on all aspects.

Format your response as:
Answer A: Accuracy=X, Relevance=X, Evidence=X, Total=XX
Answer B: Accuracy=X, Relevance=X, Evidence=X, Total=XX

Then on the LAST line, give the probability that B is preferred:
SCORE: X.XX
(0.00 = A clearly better, 0.50 = equal, 1.00 = B clearly better)
Base this on the total scores — larger gaps → more extreme scores.""",
    "fewshot": None,
    "max_tokens": 1024,
}

EXPERIMENTS["deep_analysis_sonnet"] = {
    "model": SONNET_MODEL,
    "system": """\
You are an expert at predicting which answer human evaluators preferred in a QA comparison dataset.

Perform a THOROUGH analysis:

1. QUESTION UNDERSTANDING: What exactly is being asked? What would constitute a correct answer?

2. ANSWER A ANALYSIS:
   - Key claims made
   - Factual accuracy of each claim
   - Does it answer the question?
   - Quality of references

3. ANSWER B ANALYSIS:
   - Key claims made
   - Factual accuracy of each claim
   - Does it answer the question?
   - Quality of references

4. COMPARISON: Which answer better serves someone who asked this question?

On the LAST line, give your probability estimate:
SCORE: X.XX
(0.00 = A clearly preferred, 0.50 = equal, 1.00 = B clearly preferred)""",
    "fewshot": None,
    "max_tokens": 2048,
}

EXPERIMENTS["fewshot8_prob_sonnet"] = {
    "model": SONNET_MODEL,
    "system": """\
You are predicting human preferences for QA answer comparisons. Estimate the probability that human raters preferred Answer B.

Key factors humans consider:
1. Factual correctness (most important)
2. Actually answering the question asked
3. Citation quality
4. Completeness without padding

Empty answers always lose.

Output format — on the LAST line ONLY:
SCORE: X.XX""",
    "fewshot": build_fewshot_prob(8),
    "max_tokens": 1024,
}

EXPERIMENTS["fewshot8_cot_prob_sonnet"] = {
    "model": SONNET_MODEL,
    "system": """\
You predict human preferences for QA comparisons. Analyze carefully, then estimate probability humans preferred B.

Think about:
1. Is either answer factually wrong? (biggest factor)
2. Does each answer address the actual question?
3. Are claims supported by references?
4. Is either answer empty? (automatic loss)

Reason briefly, then on the LAST line:
SCORE: X.XX
(0.00=A preferred, 0.50=equal, 1.00=B preferred)""",
    "fewshot": build_fewshot_prob(8),
    "max_tokens": 1024,
}

EXPERIMENTS["twostep_sonnet"] = {
    "model": SONNET_MODEL,
    "system": """\
You are predicting human preferences. Follow these steps exactly:

STEP 1 - Identify the correct answer to the question (1-2 sentences).
STEP 2 - Check Answer A against the correct answer. Note errors.
STEP 3 - Check Answer B against the correct answer. Note errors.
STEP 4 - Count errors and assess quality for each.
STEP 5 - Give your final estimate.

LAST line format:
SCORE: X.XX (0.00=A better, 0.50=equal, 1.00=B better)""",
    "fewshot": None,
    "max_tokens": 1536,
}

EXPERIMENTS["fact_check_sonnet"] = {
    "model": SONNET_MODEL,
    "system": """\
TASK: Predict which answer human evaluators preferred.

Your analysis MUST include:
1. State what the correct/expected answer to the question is
2. For Answer A: Is it correct? Quote the specific claim and assess it.
3. For Answer B: Is it correct? Quote the specific claim and assess it.
4. Which one would a person find more helpful?

Human raters most heavily penalized: wrong answers, irrelevant answers, empty answers.
Human raters rewarded: correct answers with good citations.

LAST line:
SCORE: X.XX (0=A better, 1=B better)""",
    "fewshot": None,
    "max_tokens": 1536,
}

EXPERIMENTS["adversarial_check_sonnet"] = {
    "model": SONNET_MODEL,
    "system": """\
You are a fact-checker evaluating two answers. Your goal is to find MISTAKES.

For each answer:
- Try to find factual errors
- Check if the answer actually addresses the question asked
- Note if the answer is empty/missing (= automatic fail)
- Check if cited references support the claims

The answer with MORE mistakes is WORSE.
Many answers look good on the surface but contain subtle errors — look carefully.

After your analysis, on the LAST line:
SCORE: X.XX (0.00 = A is better / fewer errors, 1.00 = B is better / fewer errors)""",
    "fewshot": None,
    "max_tokens": 1536,
}

EXPERIMENTS["calibrated_fewshot_sonnet"] = {
    "model": SONNET_MODEL,
    "system": """\
Predict which answer humans preferred. Use a calibrated probability.

Guidelines for calibration:
- 0.0-0.1: A is clearly and obviously correct, B is wrong or empty
- 0.2-0.3: A is better but B has some merit
- 0.4-0.6: Roughly equal quality, or both flawed/both good
- 0.7-0.8: B is better but A has some merit
- 0.9-1.0: B is clearly and obviously correct, A is wrong or empty

Focus on: correctness, relevance, citations, completeness.

LAST line only:
SCORE: X.XX""",
    "fewshot": build_fewshot_prob(8),
    "max_tokens": 1024,
}


async def main():
    client = anthropic.AsyncAnthropic()

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} prior results\n")
    else:
        all_results = {}

    for name, cfg in EXPERIMENTS.items():
        if name in all_results and all_results[name].get("corr") is not None:
            r = all_results[name]
            print(f"[{name}] (cached) cal={r['cal_ratio']:.3f} corr={r['corr']:.3f}")
            continue

        print(f"\nTesting: {name}")
        t0 = time.time()
        try:
            preds = await get_preds(
                client, cfg["model"], cfg["system"],
                label=name, fewshot=cfg.get("fewshot"),
                max_tokens=cfg.get("max_tokens", 1024),
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
        print(f"  (saved, {time.time()-t0:.0f}s)")

    # Ensembles
    print(f"\n{'='*70}")
    print("ENSEMBLES")
    print(f"{'='*70}")

    pred_arrays = {}
    for name, r in all_results.items():
        if "predictions" in r and r.get("corr") is not None:
            arr = np.array([v if v is not None else np.nan for v in r["predictions"]])
            if np.sum(~np.isnan(arr)) > SAMPLE_SIZE * 0.8:
                pred_arrays[name] = arr

    if len(pred_arrays) >= 2:
        # Also load round 3 best predictions
        r3_path = "data/round3_results.json"
        if os.path.exists(r3_path):
            with open(r3_path) as f:
                r3 = json.load(f)
            for name, r in r3.items():
                if "predictions" in r and r.get("corr") is not None:
                    arr = np.array([v if v is not None else np.nan for v in r["predictions"]])
                    if np.sum(~np.isnan(arr)) > SAMPLE_SIZE * 0.8:
                        pred_arrays[f"r3_{name}"] = arr

        sorted_by_corr = sorted(pred_arrays.keys(),
            key=lambda k: all_results.get(k, r3.get(k.replace("r3_", ""), {})).get("corr", 0),
            reverse=True)

        # Top-5 by correlation across all rounds
        top5 = sorted_by_corr[:5]
        print(f"  Top-5 by corr (all rounds): {top5}")
        ens5 = np.nanmean(np.stack([pred_arrays[k] for k in top5]), axis=0)
        m = evaluate(Y, ens5, label="top5_all_rounds")
        all_results["ensemble_top5_all"] = m

        # All R4 ensemble
        r4_names = [k for k in pred_arrays if not k.startswith("r3_")]
        if len(r4_names) >= 2:
            ens_r4 = np.nanmean(np.stack([pred_arrays[k] for k in r4_names]), axis=0)
            m = evaluate(Y, ens_r4, label="ensemble_all_r4")
            all_results["ensemble_all_r4"] = m

        # Grand ensemble (everything)
        all_arrs = list(pred_arrays.values())
        grand = np.nanmean(np.stack(all_arrs), axis=0)
        m = evaluate(Y, grand, label=f"grand_ensemble_{len(all_arrs)}")
        all_results["grand_ensemble"] = m

        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

    # Final ranking
    print(f"\n{'='*70}")
    print(f"FINAL RANKING (target cal_ratio = 0.500)")
    print(f"{'='*70}")
    valid = {k: v for k, v in all_results.items()
             if v.get("cal_ratio") is not None and v.get("corr") is not None}
    ranking = sorted(valid, key=lambda k: valid[k]["cal_ratio"])
    print(f"{'Name':45s} {'raw':>8s} {'cal':>8s} {'corr':>8s}")
    print("-" * 72)
    for name in ranking[:15]:
        r = valid[name]
        print(f"  {name:43s} {r['raw_ratio']:8.3f} {r['cal_ratio']:8.3f} {r['corr']:8.3f}")

    best = ranking[0]
    print(f"\nBest: {best}")
    print(f"  Cal ratio   = {valid[best]['cal_ratio']:.3f}")
    print(f"  Correlation = {valid[best]['corr']:.3f}")
    print(f"  Target      = 0.500 (needs corr ≈ 0.71)")


if __name__ == "__main__":
    asyncio.run(main())
