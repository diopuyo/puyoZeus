#!/bin/bash
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

# baseline / cycle32d / cycle32e / cycle32g (v89m3 のみ)
for cycle in baseline cycle32d cycle32e cycle32g; do
  if [ -f "logs/board_logs/${cycle}_v89m3.jsonl" ]; then
    PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
      --board-log "logs/board_logs/${cycle}_v89m3.jsonl" \
      --report-out "data/verify/retrospective_eval/${cycle}_v89m3.json" \
      > /dev/null 2>&1
    echo "[reeval] ${cycle} (v89m3)"
  fi
done

# cycle 33-36 (3 動画)
for cycle in cycle33 cycle34 cycle35 cycle36; do
  for vid in v89m3 v97m11 v70m2; do
    if [ -f "logs/board_logs/${cycle}_${vid}.jsonl" ]; then
      PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
        --board-log "logs/board_logs/${cycle}_${vid}.jsonl" \
        --report-out "data/verify/${cycle}_eval/${cycle}_${vid}.json" \
        > /dev/null 2>&1
      echo "[reeval] ${cycle} (${vid})"
    fi
  done
done

echo "=== ALL REEVAL DONE ==="
PYTHONPATH=. ./venv/bin/python -m scripts._summarize_c33_36
