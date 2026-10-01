#!/bin/bash
# 全WSLの計算数と空き容量を確認し、切断後もジョブを継続する。
set -euo pipefail
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
AVAILABLE=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
(( AVAILABLE >= 6291456 )) || exit 3
COUNT=$(pgrep -fc '^/mnt/.*/venv/bin/python' || true)
(( COUNT < 8 )) || exit 4
mkdir -p logs/prefire_prediction/v6
JOB=$1
shift
LOG=logs/prefire_prediction/v6/$JOB.log
[[ ! -e "$LOG" ]] || exit 5
setsid -f bash -c '"$0" -B "$@"; CODE=$?; echo EXIT_CODE=$CODE; exit $CODE' "$PY" "$@" > "$LOG" 2>&1 < /dev/null
