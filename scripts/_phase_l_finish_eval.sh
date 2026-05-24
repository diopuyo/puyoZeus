#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
for k in v75m14 v89m7 v95m3; do
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "logs/phase_l/viz_${k}.jsonl" \
    --report-out "data/verify/phase_l_eval/phase_l_${k}.json" \
    > "logs/phase_l/eval_${k}.log" 2>&1
  echo "[done] $k"
done
ls data/verify/phase_l_eval/
echo "=== Phase L FULL DONE @ $(date) ===" | tee logs/phase_l/all_done.flag
