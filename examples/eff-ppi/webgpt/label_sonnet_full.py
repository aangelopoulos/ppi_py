"""Label the full WebGPT dataset with Sonnet. Checkpoint-safe."""

import asyncio
import json
import os
import re
import time
import numpy as np
import pandas as pd
import anthropic

MODEL = "claude-4-sonnet-20250514"
MAX_CONCURRENT = 40
INPUT_PATH = "data/webgpt_comparisons.csv"
OUTPUT_PATH = "data/webgpt_sonnet_full.csv"
SEED = 42

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

df = pd.read_csv(INPUT_PATH)
N_TARGET = len(df)

# Few-shot from a held-out slice (rows 5000-5008)
np.random.seed(SEED + 1)
fs_idx = np.random.choice(np.arange(5000, min(5100, N_TARGET)), size=8, replace=False)
fs_df = df.iloc[fs_idx].reset_index(drop=True)

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
    # Mark the last few-shot assistant message for caching
    content = f"SCORE: {r['vote']:.2f}"
    if i == len(fs_df) - 1:
        fewshot.append({
            "role": "assistant",
            "content": [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}],
        })
    else:
        fewshot.append({"role": "assistant", "content": content})

def parse(text):
    for line in reversed(text.strip().upper().split("\n")[-5:]):
        m = re.search(r"SCORE[:\s=]*([01]\.?\d*)", line)
        if m:
            v = float(m.group(1))
            if 0 <= v <= 1: return v
    return None

# Load checkpoint
fwd = {}
rev = {}
if os.path.exists(OUTPUT_PATH):
    df_existing = pd.read_csv(OUTPUT_PATH)
    if "yhat_fwd" in df_existing.columns:
        for i in range(min(len(df_existing), N_TARGET)):
            if pd.notna(df_existing.loc[i, "yhat_fwd"]):
                fwd[i] = df_existing.loc[i, "yhat_fwd"]
            if pd.notna(df_existing.loc[i, "yhat_rev"]):
                rev[i] = df_existing.loc[i, "yhat_rev"]
    elif "yhat" in df_existing.columns:
        for i in range(min(len(df_existing), N_TARGET)):
            if pd.notna(df_existing.loc[i, "yhat"]):
                fwd[i] = df_existing.loc[i, "yhat"]

n_done_fwd = len(fwd)
n_done_rev = len(rev)
print(f"Target: {N_TARGET} rows | Checkpoint: {n_done_fwd} fwd, {n_done_rev} rev")

async def main():
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    credits_dead = asyncio.Event()

    async def go(i, row, is_fwd):
        if credits_dead.is_set(): return
        async with sem:
            if credits_dead.is_set(): return
            if is_fwd:
                msg = make_msg(row["question"], row["answer_0"], row["answer_1"])
            else:
                msg = make_msg(row["question"], row["answer_1"], row["answer_0"])
            msgs = list(fewshot) + [{"role": "user", "content": msg}]
            for attempt in range(3):
                if credits_dead.is_set(): return
                try:
                    r = await client.messages.create(
                        model=MODEL, max_tokens=1024,
                        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                        messages=msgs)
                    raw = parse(r.content[0].text)
                    if raw is None:
                        if attempt < 2: continue
                        return
                    if is_fwd: fwd[i] = raw
                    else: rev[i] = 1.0 - raw
                    return
                except anthropic.RateLimitError:
                    await asyncio.sleep(min(2 ** attempt, 4))
                except anthropic.APIStatusError as e:
                    if "credit" in str(e).lower():
                        credits_dead.set()
                    return
                except: return

    def save_checkpoint():
        yhat_fwd_arr = [fwd.get(i, np.nan) for i in range(N_TARGET)]
        yhat_rev_arr = [rev.get(i, np.nan) for i in range(N_TARGET)]
        yhat_arr = []
        for i in range(N_TARGET):
            vals = [v for v in [fwd.get(i), rev.get(i)] if v is not None]
            yhat_arr.append(np.mean(vals) if vals else np.nan)
        df_out = df.copy()
        df_out["yhat_fwd"] = yhat_fwd_arr
        df_out["yhat_rev"] = yhat_rev_arr
        df_out["yhat"] = yhat_arr
        df_out.to_csv(OUTPUT_PATH, index=False)
        return yhat_arr

    batch_size = 200
    t0 = time.time()
    for batch_start in range(0, N_TARGET, batch_size):
        batch_end = min(batch_start + batch_size, N_TARGET)
        tasks = []
        for i in range(batch_start, batch_end):
            row = df.iloc[i]
            if i not in fwd:
                tasks.append(go(i, row, True))
            if i not in rev:
                tasks.append(go(i, row, False))
        if not tasks:
            continue

        await asyncio.gather(*tasks)

        yhat_arr = save_checkpoint()

        n_fwd = len(fwd)
        n_rev = len(rev)
        n_both = sum(1 for i in range(N_TARGET) if i in fwd and i in rev)
        elapsed = time.time() - t0
        calls_done = (n_fwd - n_done_fwd) + (n_rev - n_done_rev)
        rate = max(calls_done, 1) / elapsed
        print(f"  fwd={n_fwd} rev={n_rev} both={n_both}/{N_TARGET} "
              f"({n_both/N_TARGET*100:.1f}%) | {rate:.1f} calls/s")

        if credits_dead.is_set():
            print("Credits exhausted. Checkpoint saved.")
            break

    # Final summary
    yhat = np.array(yhat_arr)
    mask = ~np.isnan(yhat)
    Y = df.loc[mask, "vote"].values
    Yhat = yhat[mask]

    if mask.sum() < 50:
        print(f"\nOnly {mask.sum()} rows labeled. Need more credits.")
        return

    var_y = np.var(Y)
    var_resid = np.var(Y - Yhat)
    corr = np.corrcoef(Y, Yhat)[0, 1]

    print(f"\n{'='*50}")
    print(f"Results ({mask.sum()}/{N_TARGET} labeled):")
    print(f"  Var(Y)          = {var_y:.6f}")
    print(f"  Var(Y-Yhat)     = {var_resid:.6f}  ratio={var_resid/var_y:.3f}")
    print(f"  Corr(Y, Yhat)   = {corr:.4f}")
    print(f"{'='*50}")

asyncio.run(main())
