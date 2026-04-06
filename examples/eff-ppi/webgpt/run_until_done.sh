#!/bin/bash
# Repeatedly run pseudolabel_final.py, waiting between runs for credits to refresh.
# Checkpoint-safe: each run picks up where the last left off.

MAX_ATTEMPTS=50
WAIT_SECONDS=180

for i in $(seq 1 $MAX_ATTEMPTS); do
    echo "=== Attempt $i / $MAX_ATTEMPTS ($(date)) ==="

    # Check if already done
    DONE=$(python -c "
import pandas as pd
try:
    df = pd.read_csv('data/webgpt_pseudolabels_final.csv')
    n = df.yhat.notna().sum()
    print(f'{n}/{len(df)}')
    if n >= len(df) * 0.95:
        exit(0)
    else:
        exit(1)
except:
    exit(1)
" 2>/dev/null)

    if [ $? -eq 0 ]; then
        echo "Done! $DONE rows labeled."
        break
    fi
    echo "Progress: $DONE"

    # Check credits
    CREDITS=$(python -c "
import anthropic
c = anthropic.Anthropic()
try:
    r = c.messages.create(model='claude-haiku-4-5-20251001', max_tokens=5, messages=[{'role':'user','content':'hi'}])
    print('OK')
except:
    print('NO')
" 2>/dev/null)

    if [ "$CREDITS" != "OK" ]; then
        echo "No credits. Waiting ${WAIT_SECONDS}s..."
        sleep $WAIT_SECONDS
        continue
    fi

    echo "Credits OK. Running..."
    python pseudolabel_final.py
    echo "Run finished. Waiting ${WAIT_SECONDS}s for credits to refresh..."
    sleep $WAIT_SECONDS
done

echo "=== Final status ==="
python -c "
import pandas as pd
df = pd.read_csv('data/webgpt_pseudolabels_final.csv')
n = df.yhat.notna().sum()
print(f'{n}/{len(df)} rows labeled ({n/len(df)*100:.1f}%)')
" 2>/dev/null
