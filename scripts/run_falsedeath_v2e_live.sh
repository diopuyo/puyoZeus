#!/bin/bash
# 統合側の実動画手順を読み取り実行し、新規出力はevalset内だけに置く。
set -eu
ROOT=/mnt/d/puyo_analyzer/wt_evalset
OUT="$ROOT/logs/eval_set/falsedeath_fix/v2e/live_with_assets"
mkdir -p "$OUT"
test ! -e "$OUT/run.log"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export H1_PRIME_OUTPUT="$OUT/audit"
source "$ROOT/logs/resource_falsedeath.sh"
cd "$ROOT/logs/eval_set/falsedeath_fix/v2e/live_runtime"
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
run_limited "$OUT/run.log" nice -n 10 "$PY" -B "$ROOT/scripts/verify_falsedeath_v2e_live.py" \
  --source video --realtime --start-sec 2579 --end-sec 2650 --warmup-sec 1 \
  --cpu-threads 1 --evaluation-nice 10 --cnn-device cpu --recognition-audit --runtime-audit \
  --output "$OUT" --port 8877
