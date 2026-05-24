#!/usr/bin/env bash
# v91 で 5/10/20/30/40/50/60/70s 8 frame に diag を実行 → 検出率推移
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
mkdir -p logs data/diag_results
for sec in 5 10 20 30 40 50 60 70; do
    PYTHONPATH=. ./venv/bin/python -m scripts.diagnose_cell_classification \
        --video data/test_unknown/v91_match1_75s_720p.mp4 \
        --sec "$sec" \
        > "data/diag_results/v91_${sec}s_FIX_O.txt" 2>&1
done
echo done
