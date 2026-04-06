"""Generate pseudolabels for WebGPT comparisons using Claude Sonnet 4.6."""

import asyncio
import json
import os
import time

import anthropic
import numpy as np
import pandas as pd

MODEL = "claude-4-sonnet-20250514"
MAX_CONCURRENT = 50
INPUT_PATH = "data/webgpt_comparisons.csv"
OUTPUT_PATH = "data/webgpt_pseudolabels.csv"

SYSTEM_PROMPT = """\
You are an expert evaluator of answer quality. Given a question and two candidate answers, \
judge which answer is better based on accuracy, relevance, completeness, and clarity.

Output a single number between 0 and 1:
- 0.0 = Answer 0 is clearly better
- 0.25 = Answer 0 is somewhat better
- 0.5 = Both answers are about equal
- 0.75 = Answer 1 is somewhat better
- 1.0 = Answer 1 is clearly better

Consider:
1. Factual correctness — does the answer contain accurate information?
2. Relevance — does the answer actually address the question?
3. Completeness — does the answer provide sufficient detail?
4. Clarity — is the answer well-written and easy to understand?
5. Citations — are claims supported by references?

An empty or missing answer is always worse than a substantive one.

Output ONLY the number, nothing else."""

USER_TEMPLATE = """\
Question: {question}

Answer 0: {answer_0}

Answer 1: {answer_1}"""


async def get_pseudolabel(client, semaphore, idx, row):
    """Get a pseudolabel for a single row."""
    async with semaphore:
        user_msg = USER_TEMPLATE.format(
            question=row["question"],
            answer_0=row["answer_0"] if pd.notna(row["answer_0"]) else "(no answer provided)",
            answer_1=row["answer_1"] if pd.notna(row["answer_1"]) else "(no answer provided)",
        )
        for attempt in range(5):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=16,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                import re
                text = response.content[0].text.strip()
                match = re.match(r"^([\d.]+)", text)
                if not match:
                    raise ValueError(f"No number found in '{text[:50]}'")
                score = float(match.group(1))
                if not (0 <= score <= 1):
                    raise ValueError(f"Score {score} out of range")
                return idx, score
            except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
                if attempt < 4:
                    wait = 2 ** (attempt + 1)
                    await asyncio.sleep(wait)
                else:
                    print(f"Row {idx}: API error after retries: {e}")
                    return idx, None
            except (ValueError, IndexError) as e:
                if attempt < 4:
                    continue
                print(f"Row {idx}: Parse error '{text}': {e}")
                return idx, None


async def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    # Resume from partial results if available
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        done_indices = set(existing.index[existing["yhat"].notna()])
        print(f"Resuming: {len(done_indices)} already done")
    else:
        existing = None
        done_indices = set()

    todo = [(i, row) for i, row in df.iterrows() if i not in done_indices]
    print(f"Processing {len(todo)} rows...")

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Process in batches and save periodically
    results = {}
    if existing is not None:
        for i, val in existing["yhat"].items():
            if pd.notna(val):
                results[i] = val

    batch_size = 500
    t0 = time.time()
    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start : batch_start + batch_size]
        tasks = [get_pseudolabel(client, semaphore, idx, row) for idx, row in batch]
        batch_results = await asyncio.gather(*tasks)

        for idx, score in batch_results:
            if score is not None:
                results[idx] = score

        elapsed = time.time() - t0
        done = batch_start + len(batch)
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  {done}/{len(todo)} done ({rate:.1f} rows/s)")

        # Save checkpoint
        df["yhat"] = pd.Series(results)
        df.to_csv(OUTPUT_PATH, index=False)

    # Final save
    df["yhat"] = pd.Series(results)
    df.to_csv(OUTPUT_PATH, index=False)

    # Compute variance metrics
    mask = df["yhat"].notna()
    Y = df.loc[mask, "vote"].values
    Yhat = df.loc[mask, "yhat"].values

    var_Y = np.var(Y)
    var_residual = np.var(Y - Yhat)

    print(f"\n{'='*40}")
    print(f"Results ({mask.sum()} labeled out of {len(df)}):")
    print(f"  Var(Y)       = {var_Y:.6f}")
    print(f"  Var(Y - Yhat)= {var_residual:.6f}")
    print(f"  Ratio        = {var_residual / var_Y:.4f}")
    print(f"  (Lower ratio = better pseudolabels)")
    print(f"{'='*40}")


if __name__ == "__main__":
    asyncio.run(main())
