"""Download and process the WebGPT comparisons dataset."""

import json
import os
import urllib.request
import pandas as pd

DATA_URL = "https://openaipublic.blob.core.windows.net/webgpt-answer-viewer/comparisons.jsonl"
RAW_PATH = "data/comparisons.jsonl"
OUT_PATH = "data/webgpt_comparisons.csv"

# Download raw data if not already present
if not os.path.exists(RAW_PATH):
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, RAW_PATH)
    print("Download complete.")

# Load JSONL — each line is a pair [item_0, item_1]
rows = []
with open(RAW_PATH) as f:
    for line in f:
        pair = json.loads(line)
        question = pair[0]["question"]["full_text"]
        answer_0 = pair[0]["answer"]
        answer_1 = pair[1]["answer"]
        score_0 = pair[0]["score"]
        # Remap from [-1, 1] (answer_0 preferred to answer_1 preferred) to [0, 1]
        vote = (1 - score_0) / 2
        rows.append({"question": question, "answer_0": answer_0, "answer_1": answer_1, "vote": vote})

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)
print(f"Saved {len(df)} rows to {OUT_PATH}")
print(df.head())
print(f"\nVote distribution:\n{df['vote'].value_counts().sort_index()}")
