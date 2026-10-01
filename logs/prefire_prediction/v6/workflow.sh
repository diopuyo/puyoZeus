#!/bin/bash
# 他ジョブ終了後の正式測定から、学習・採点まで順番を固定する。
set -euo pipefail
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
OUT=logs/prefire_prediction/v6
test -f "$OUT/CHECKS_v2.json"
"$PY" -B -m scripts.prefire_v6_measure benchmark > "$OUT/benchmark_isolated.log" 2>&1
"$PY" -B -m scripts.prefire_v6_batch > "$OUT/batch.log" 2>&1
echo completed
