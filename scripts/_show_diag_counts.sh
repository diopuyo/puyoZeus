#!/usr/bin/env bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
for sec in 5 10 20 30 40 50 60 70; do
    f="data/diag_results/v91_${sec}s_FIX_O.txt"
    if [ -f "$f" ]; then
        line=$(grep 'count' "$f" | head -2 | tr '\n' '|')
        echo "v91 ${sec}s: ${line}"
    fi
done
