#!/bin/bash
# A2専用。全計算を共通の資源制限と低優先度で実行する。
set -eu
cd /mnt/d/puyo_analyzer/wt_a2
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT="$PWD/logs/a2_safety"
export FALSEDEATH_POLL_SECONDS="${FALSEDEATH_POLL_SECONDS:-2}"
mkdir -p "$OUT"
source scripts/a2_resource_limits.sh
name="$1"
shift
run_limited "$OUT/$name.log" nice -n 10 <home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -B "$@"
