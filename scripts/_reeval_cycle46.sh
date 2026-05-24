#!/bin/bash
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
for v in v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11; do
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "logs/board_logs/cycle46_${v}_buf15.jsonl" \
    --report-out "data/verify/cycle46_eval/cycle46_${v}_buf15.json" \
    > /dev/null 2>&1
  echo "[reeval] cycle46 $v"
done
PYTHONPATH=. ./venv/bin/python -m scripts._summarize_chain_metrics
