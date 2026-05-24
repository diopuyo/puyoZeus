#!/bin/bash
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
for c in cycle37 cycle38 cycle39; do
  for v in v89m3 v97m11 v70m2; do
    PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
      --board-log "logs/board_logs/${c}_${v}.jsonl" \
      --report-out "data/verify/${c}_eval/${c}_${v}.json" \
      > /dev/null 2>&1
  done
done
echo done
PYTHONPATH=. ./venv/bin/python -m scripts._summarize_c33_37
