"""Optimize the pseudolabeling prompt using Haiku on a small sample.

Robust to interruption — saves results after each prompt evaluation.
Re-run to continue where you left off.
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
MAX_CONCURRENT = 50
INPUT_PATH = "data/webgpt_comparisons.csv"
RESULTS_PATH = "data/prompt_optimization_results.json"
SAMPLE_SIZE = 300
SEED = 42

# ============================================================
# Data setup
# ============================================================
df_full = pd.read_csv(INPUT_PATH)
np.random.seed(SEED)
sample_idx = np.random.choice(len(df_full), size=SAMPLE_SIZE, replace=False)
df_sample = df_full.iloc[sample_idx].reset_index(drop=True)
Y = df_sample["vote"].values
VAR_Y = float(np.var(Y))
TARGET = VAR_Y / 2

print(f"Sample: {SAMPLE_SIZE} rows | Var(Y) = {VAR_Y:.6f} | Target Var(Y-Yhat) ≈ {TARGET:.6f}")
print()

# ============================================================
# Prompt definitions
# ============================================================

# Each prompt returns answer in {A, B, TIE} which we map to scores.
# To remove positional bias, we run BOTH orderings and average.

def make_user_msg(question, ans_first, ans_second):
    a0 = ans_first if pd.notna(ans_first) else "(no answer provided)"
    a1 = ans_second if pd.notna(ans_second) else "(no answer provided)"
    return f"Question: {question}\n\nAnswer A: {a0}\n\nAnswer B: {a1}"


def parse_ab_verdict(text):
    """Robustly parse A/B/TIE from model output. Returns score in [0,1]."""
    text_upper = text.strip().upper()
    last_line = text_upper.split("\n")[-1].strip()

    # Check for confidence-graded verdicts first (v9 style)
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

    # Standard A/B/TIE patterns
    for line in [last_line, text_upper]:
        m = re.search(r"(?:BETTER|WINNER|VERDICT|ANSWER)[:\s]*([AB]|TIE)\b", line)
        if m:
            v = m.group(1)
            return 0.0 if v == "A" else 1.0 if v == "B" else 0.5
        if re.match(r"^[AB]$", line):
            return 0.0 if line == "A" else 1.0
        if "TIE" in line:
            return 0.5

    # Fallback
    m = re.search(r"\b(ANSWER\s*[AB])\b", text_upper)
    if m:
        return 0.0 if m.group(1)[-1] == "A" else 1.0

    if text_upper.startswith("A"):
        return 0.0
    if text_upper.startswith("B"):
        return 1.0
    return None


PROMPTS = {}

PROMPTS["v1_simple_ab"] = """\
Which answer better addresses the question? Consider accuracy, relevance, and completeness.
An empty or missing answer always loses.

Reply with ONLY: A, B, or TIE"""

PROMPTS["v2_cot_verdict"] = """\
You are an expert answer evaluator. Compare two candidate answers to a question.

Think step by step:
1. Does each answer actually address the question?
2. Is the information factually correct?
3. How complete and well-supported is each answer?
4. An empty or missing answer always loses.

After your reasoning, write your final verdict on the LAST line as exactly one of:
VERDICT: A
VERDICT: B
VERDICT: TIE"""

PROMPTS["v3_accuracy_focus"] = """\
Which answer contains more accurate, factually correct information for the question?
Ignore style — focus only on correctness and whether the question is actually answered.
An empty or missing answer always loses.

Reply with ONLY: A, B, or TIE"""

PROMPTS["v4_human_pref"] = """\
Imagine a person asked this question online. Which answer would they find more helpful?
Consider: Does it answer their question correctly? Is it clear? Is it well-supported?
An empty or missing answer always loses.

Reply with ONLY: A, B, or TIE"""

PROMPTS["v5_detailed_cot"] = """\
You are predicting which answer a human evaluator would prefer.

Evaluate each answer on:
- ACCURACY: Are the stated facts correct? (most important)
- RELEVANCE: Does it answer the actual question asked?
- EVIDENCE: Does it cite sources to support claims?
- COMPLETENESS: Does it cover the topic adequately without excessive padding?

An empty or missing answer is always worse.

Reason briefly about each criterion, then give your final verdict on the last line:
VERDICT: A (if Answer A is better)
VERDICT: B (if Answer B is better)
VERDICT: TIE (if roughly equal)"""

PROMPTS["v6_concise_strict"] = """\
Pick the better answer. Consider only: (1) factual correctness, (2) answers the question, (3) evidence/citations. Empty answer = automatic loss.

One letter only: A or B"""

PROMPTS["v7_ref_quality"] = """\
You are evaluating answers from a QA system. Each answer may include references like [1], [2].

Judge which answer:
- Correctly answers the specific question asked
- Has accurate factual claims supported by references
- Doesn't include irrelevant or incorrect information

An empty or missing answer automatically loses.

Reply with your verdict on the last line:
VERDICT: A
VERDICT: B
VERDICT: TIE"""

PROMPTS["v8_calibrated"] = """\
You are calibrating human preference predictions for a dataset of question-answer comparisons.

In this dataset, Answer A is preferred about 50% of the time and Answer B about 50% of the time. Your job is to predict which one humans actually chose as better.

Humans judge based on: accuracy, relevance, helpfulness, and citation quality. An empty answer always loses.

Think briefly, then give your prediction on the last line:
VERDICT: A
VERDICT: B
VERDICT: TIE"""

# ============================================================
# Round 2: Hybrid prompts combining high-corr CoT with calibration
# ============================================================

PROMPTS["v9_cot_confidence"] = """\
You are predicting human preferences for answer quality. Evaluate both answers carefully.

Step 1: Check if either answer is empty/missing — that answer automatically loses.
Step 2: Check factual accuracy — wrong facts are heavily penalized by human raters.
Step 3: Check relevance — does each answer address the specific question asked?
Step 4: Check evidence — are claims supported by references?

After reasoning, give your verdict AND confidence on the last line:
VERDICT: A STRONG (very confident A is better)
VERDICT: A WEAK (slightly lean toward A)
VERDICT: TIE
VERDICT: B WEAK (slightly lean toward B)
VERDICT: B STRONG (very confident B is better)"""

PROMPTS["v10_cot_accuracy_first"] = """\
Compare these two answers. The MOST important criterion is factual accuracy.

Think through:
1. Is either answer empty or missing? (automatic loss)
2. Does either answer contain factually wrong information?
3. Does each answer actually address the specific question?
4. Which is better supported by references?

Then give your verdict on the LAST line:
VERDICT: A
VERDICT: B
VERDICT: TIE"""

PROMPTS["v11_simulate_human"] = """\
You are simulating a human evaluator rating answers from a QA system.

The human evaluator:
- Cares most about getting the RIGHT answer to their question
- Values clear, direct answers over verbose ones
- Appreciates when answers cite their sources
- Will mark an empty/missing answer as worse
- Sometimes thinks both answers are equally good or bad

Read both answers, then predict what the human would choose.
On the last line, write:
VERDICT: A
VERDICT: B
VERDICT: TIE"""

PROMPTS["v12_error_detection"] = """\
Your task: identify which answer has MORE errors or problems.

Check for:
- Factual mistakes or wrong claims
- Not answering the actual question asked
- Missing answer (empty = worst)
- Unsupported claims without references
- Irrelevant information that doesn't help

The answer with MORE problems is the WORSE answer.

On the last line:
VERDICT: A (if A has fewer problems = A is better)
VERDICT: B (if B has fewer problems = B is better)
VERDICT: TIE (similar quality)"""

PROMPTS["v13_webgpt_specific"] = """\
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
VERDICT: TIE"""

PROMPTS["v14_short_cot_strict"] = """\
Compare the two answers. One sentence of reasoning, then your verdict.

Criteria: correctness > relevance > evidence. Empty answer = loss.

Format (last line must be exactly one of):
VERDICT: A
VERDICT: B
VERDICT: TIE"""

PROMPTS["v15_pairwise_rubric"] = """\
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
VERDICT: TIE"""

PROMPTS["v16_tiebreak_bias"] = """\
Which answer is better? Most answer pairs have a clear winner — ties are rare.

Consider accuracy, relevance, and evidence. Empty = automatic loss.
Pick one:
VERDICT: A
VERDICT: B"""

# ============================================================
# Evaluation with positional debiasing
# ============================================================

async def evaluate_prompt(client, system_prompt, label=""):
    """Evaluate a prompt with answer-order swapping to remove positional bias."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # We'll run each example TWICE: once as (ans0, ans1) and once as (ans1, ans0)
    # Then average the scores.
    scores_fwd = [None] * SAMPLE_SIZE  # original order: A=ans0, B=ans1
    scores_rev = [None] * SAMPLE_SIZE  # swapped order: A=ans1, B=ans0

    async def process_one(i, row, is_forward):
        async with semaphore:
            if is_forward:
                msg = make_user_msg(row["question"], row["answer_0"], row["answer_1"])
            else:
                msg = make_user_msg(row["question"], row["answer_1"], row["answer_0"])

            for attempt in range(3):
                try:
                    response = await client.messages.create(
                        model=HAIKU_MODEL,
                        max_tokens=512,
                        system=system_prompt,
                        messages=[{"role": "user", "content": msg}],
                    )
                    text = response.content[0].text.strip()
                    raw = parse_ab_verdict(text)

                    if raw is None:
                        if attempt < 2:
                            continue
                        return  # give up

                    # For forward: A=ans0, B=ans1 → raw 0=ans0 better, 1=ans1 better ✓
                    # For reverse: A=ans1, B=ans0 → raw 0=ans1 better, 1=ans0 better → flip
                    if is_forward:
                        if scores_fwd[i] is None:
                            scores_fwd[i] = raw
                    else:
                        if scores_rev[i] is None:
                            scores_rev[i] = 1.0 - raw  # flip back to ans0/ans1 frame
                    return

                except anthropic.RateLimitError:
                    await asyncio.sleep(2 ** (attempt + 1))
                except anthropic.APIStatusError as e:
                    if "credit balance" in str(e):
                        raise  # propagate credit errors
                    await asyncio.sleep(2 ** (attempt + 1))
                except Exception:
                    if attempt < 2:
                        continue
                    return

    tasks = []
    for i, row in df_sample.iterrows():
        tasks.append(process_one(i, row, True))
        tasks.append(process_one(i, row, False))

    try:
        await asyncio.gather(*tasks)
    except anthropic.APIStatusError as e:
        if "credit balance" in str(e):
            print(f"  [{label}] CREDIT ERROR — partial results")

    # Average forward and reverse scores
    Yhat = np.full(SAMPLE_SIZE, np.nan)
    for i in range(SAMPLE_SIZE):
        vals = [v for v in [scores_fwd[i], scores_rev[i]] if v is not None]
        if vals:
            Yhat[i] = np.mean(vals)

    mask = ~np.isnan(Yhat)
    n_valid = mask.sum()
    n_fwd = sum(1 for v in scores_fwd if v is not None)
    n_rev = sum(1 for v in scores_rev if v is not None)

    if n_valid < SAMPLE_SIZE * 0.5:
        print(f"  [{label}] Too many failures: {n_valid}/{SAMPLE_SIZE} valid (fwd={n_fwd}, rev={n_rev})")
        return {"var_resid": None, "n_valid": int(n_valid), "label": label}

    Y_sub = Y[mask]
    Yhat_sub = Yhat[mask]
    var_resid = float(np.var(Y_sub - Yhat_sub))
    var_y = float(np.var(Y_sub))
    corr = float(np.corrcoef(Y_sub, Yhat_sub)[0, 1])
    bias = float(np.mean(Yhat_sub) - np.mean(Y_sub))
    yhat_mean = float(np.mean(Yhat_sub))
    yhat_std = float(np.std(Yhat_sub))

    ratio = var_resid / var_y if var_y > 0 else float("inf")
    print(f"  [{label}] n={n_valid} | Var(Y-Yhat)={var_resid:.6f} | ratio={ratio:.3f} | corr={corr:.3f} | bias={bias:+.3f} | Yhat μ={yhat_mean:.3f} σ={yhat_std:.3f}")

    return {
        "var_resid": var_resid, "var_y": var_y, "ratio": ratio,
        "corr": corr, "bias": bias, "yhat_mean": yhat_mean, "yhat_std": yhat_std,
        "n_valid": int(n_valid), "label": label,
    }


# ============================================================
# Main
# ============================================================

async def main():
    client = anthropic.AsyncAnthropic()

    # Load prior results
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} prior results from {RESULTS_PATH}")
    else:
        all_results = {}

    # Test each prompt that hasn't been tested yet
    for name, system_prompt in PROMPTS.items():
        if name in all_results and all_results[name].get("var_resid") is not None:
            r = all_results[name]
            print(f"[{name}] (cached) Var(Y-Yhat)={r['var_resid']:.6f} | ratio={r['ratio']:.3f} | corr={r['corr']:.3f}")
            continue

        print(f"\nTesting: {name}")
        t0 = time.time()
        try:
            result = await evaluate_prompt(client, system_prompt, label=name)
            result["system_prompt"] = system_prompt
            result["elapsed_s"] = time.time() - t0
            all_results[name] = result
        except Exception as e:
            print(f"  [{name}] Error: {e}")
            all_results[name] = {"var_resid": None, "label": name, "error": str(e)}

        # Save after each prompt
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  (checkpoint saved)")

    # Ranking
    print(f"\n{'='*70}")
    print(f"RANKING (target Var(Y-Yhat) = {TARGET:.6f})")
    print(f"{'='*70}")
    valid = {k: v for k, v in all_results.items() if v.get("var_resid") is not None}
    ranking = sorted(valid, key=lambda k: valid[k]["var_resid"])
    for i, name in enumerate(ranking):
        r = valid[name]
        print(f"  {i+1:2d}. {name:30s} Var(Y-Yhat)={r['var_resid']:.6f}  ratio={r['ratio']:.3f}  corr={r['corr']:.3f}")

    if ranking:
        best = ranking[0]
        print(f"\nBest: {best}")
        print(f"  Var(Y-Yhat) = {valid[best]['var_resid']:.6f}")
        print(f"  Target      = {TARGET:.6f}")
        print(f"  Var(Y)      = {VAR_Y:.6f}")

        # Save best prompt separately
        with open("data/best_prompt.json", "w") as f:
            json.dump({
                "name": best,
                "system_prompt": valid[best].get("system_prompt", ""),
                "var_resid": valid[best]["var_resid"],
                "var_y": VAR_Y,
                "target": TARGET,
                "ratio": valid[best]["ratio"],
                "corr": valid[best]["corr"],
            }, f, indent=2)
        print(f"  Saved to data/best_prompt.json")


if __name__ == "__main__":
    asyncio.run(main())
