#!/bin/bash
# D-v2の診断を共通リソース制御下で実行する。
set -eu
ROOT=/mnt/d/puyo_analyzer/wt_evalset
cd "$ROOT"
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
OUT="$ROOT/logs/eval_set/falsedeath_fix/v2"
mkdir -p "$OUT"
source "$ROOT/logs/resource_falsedeath.sh"
name="$1"
shift
run_limited "$OUT/$name.log" nice -n 10 "$PY" -B "$@"
