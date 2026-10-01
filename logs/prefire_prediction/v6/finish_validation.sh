#!/bin/bash
# 追加対照の終了を待ち、全体8プロセスを超えずに採点器を確認する。
set -euo pipefail
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
OUT=logs/prefire_prediction/v6
until test -f "$OUT/baseline/s1/DONE.json" && test -f "$OUT/baseline/s5/DONE.json"; do
  AVAILABLE=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  COUNT=$(pgrep -fc '^/mnt/.*/venv/bin/python' || true)
  echo "{\"time\":$(date +%s),\"mode\":\"off_control\",\"available_kib\":$AVAILABLE,\"python_count\":$COUNT}" >> "$OUT/resource_history.jsonl"
  sleep 5
done
while (( $(pgrep -fc '^/mnt/.*/venv/bin/python' || true) >= 8 )); do sleep 5; done
"$PY" -B -m pytest tests/test_prefire_v6_score.py -q > "$OUT/score_boundary_test.log" 2>&1
"$PY" -B -m scripts.prefire_v6_score > "$OUT/final_score.log" 2>&1
"$PY" -B -m scripts.prefire_v6_report > "$OUT/final_report.log" 2>&1
echo completed
