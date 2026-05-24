#!/bin/bash
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
for cycle in baseline cycle32d cycle32e cycle32g; do
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "logs/board_logs/${cycle}_v89m3.jsonl" \
    --report-out "data/verify/retrospective_eval/${cycle}_v89m3.json" \
    > /dev/null 2>&1
  echo "[done] ${cycle}"
done
PYTHONPATH=. ./venv/bin/python -m scripts._summarize_retro_eval
