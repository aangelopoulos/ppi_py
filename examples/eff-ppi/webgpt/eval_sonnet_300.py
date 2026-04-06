"""Evaluate Sonnet on 300 rows with the best prompt from optimization."""

import asyncio
import re
import numpy as np
import pandas as pd
import anthropic

MODEL = "claude-4-sonnet-20250514"
MAX_CONCURRENT = 40
SEED = 42
N = 300

df_full = pd.read_csv("data/webgpt_comparisons.csv")
np.random.seed(SEED)
idx = np.random.choice(len(df_full), size=N, replace=False)
df = df_full.iloc[idx].reset_index(drop=True)
Y = df["vote"].values

# Few-shot examples from outside sample
other = np.setdiff1d(np.arange(len(df_full)), idx)
np.random.seed(SEED + 1)
np.random.shuffle(other)
fs_df = df_full.iloc[other[:8]].reset_index(drop=True)

SYSTEM = """\
You predict human preferences for QA comparisons. Analyze carefully, then estimate probability humans preferred B.

Think about:
1. Is either answer factually wrong? (biggest factor)
2. Does each answer address the actual question?
3. Are claims supported by references?
4. Is either answer empty? (automatic loss)

Reason briefly, then on the LAST line:
SCORE: X.XX
(0.00=A preferred, 0.50=equal, 1.00=B preferred)"""

def make_msg(q, a, b):
    a = a if pd.notna(a) else "(no answer provided)"
    b = b if pd.notna(b) else "(no answer provided)"
    return f"Question: {q}\n\nAnswer A: {a}\n\nAnswer B: {b}"

fewshot = []
for i in range(len(fs_df)):
    r = fs_df.iloc[i]
    a0 = r["answer_0"] if pd.notna(r["answer_0"]) else "(no answer provided)"
    a1 = r["answer_1"] if pd.notna(r["answer_1"]) else "(no answer provided)"
    fewshot.append({"role": "user", "content": make_msg(r["question"], a0, a1)})
    fewshot.append({"role": "assistant", "content": f"SCORE: {r['vote']:.2f}"})

def parse(text):
    for line in reversed(text.strip().upper().split("\n")[-5:]):
        m = re.search(r"SCORE[:\s=]*([01]\.?\d*)", line)
        if m:
            v = float(m.group(1))
            if 0 <= v <= 1: return v
    return None

async def main():
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    fwd = [None] * N
    rev = [None] * N

    async def go(i, row, is_fwd):
        async with sem:
            if is_fwd:
                msg = make_msg(row["question"], row["answer_0"], row["answer_1"])
            else:
                msg = make_msg(row["question"], row["answer_1"], row["answer_0"])
            msgs = list(fewshot) + [{"role": "user", "content": msg}]
            for attempt in range(3):
                try:
                    r = await client.messages.create(
                        model=MODEL, max_tokens=1024, system=SYSTEM, messages=msgs)
                    raw = parse(r.content[0].text)
                    if raw is None:
                        if attempt < 2: continue
                        return
                    if is_fwd: fwd[i] = raw
                    else: rev[i] = 1.0 - raw
                    return
                except anthropic.RateLimitError:
                    await asyncio.sleep(2 ** (attempt + 1))
                except anthropic.APIStatusError as e:
                    if "credit" in str(e).lower():
                        return
                    return
                except: return

    tasks = []
    for i, row in df.iterrows():
        tasks.append(go(i, row, True))
        tasks.append(go(i, row, False))
    await asyncio.gather(*tasks)

    Yhat = np.array([np.mean([v for v in [fwd[i], rev[i]] if v is not None])
                      if any(v is not None for v in [fwd[i], rev[i]]) else np.nan
                      for i in range(N)])

    mask = ~np.isnan(Yhat)
    Ys, Yhs = Y[mask], Yhat[mask]
    var_y = np.var(Ys)
    var_resid = np.var(Ys - Yhs)
    corr = np.corrcoef(Ys, Yhs)[0, 1]
    cov = np.cov(Ys, Yhs)[0, 1]
    alpha = cov / np.var(Yhs)
    Yh_cal = np.mean(Ys) + alpha * (Yhs - np.mean(Yhs))
    var_cal = np.var(Ys - Yh_cal)

    print(f"\nResults ({mask.sum()}/{N} valid):")
    print(f"  Var(Y)              = {var_y:.6f}")
    print(f"  Var(Y-Yhat) raw     = {var_resid:.6f}  ratio={var_resid/var_y:.3f}")
    print(f"  Var(Y-Yhat) calib   = {var_cal:.6f}  ratio={var_cal/var_y:.3f}")
    print(f"  Corr(Y, Yhat)       = {corr:.4f}")
    print(f"  Yhat mean={np.mean(Yhs):.3f} std={np.std(Yhs):.3f}")
    print(f"  1-R²                = {1-corr**2:.4f}")

asyncio.run(main())
