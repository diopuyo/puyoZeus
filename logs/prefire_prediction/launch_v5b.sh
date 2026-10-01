#!/bin/bash
# 5B専用の起動口。旧原票を保持し、空きメモリと当検証の並列数を確認する。
set -euo pipefail
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
[[ "${1:-}" == run ]] || exit 2
shift
AVAILABLE=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
(( AVAILABLE >= 5242880 )) || { echo 'available memory < 5GiB (1GiB reserve)'; exit 3; }
COUNT=$(pgrep -fc '[p]ython.*scripts.prefire_v5b' || true)
(( COUNT < 6 )) || exit 4
mkdir -p logs/prefire_prediction/v5b
JOB=$1
shift
LOG=logs/prefire_prediction/v5b/$JOB.log
[[ ! -e "$LOG" ]] || { echo 'existing log'; exit 5; }
setsid -f bash -c 'exec nice -n 10 "$0" -B "$@"' "$PY" "$@" > "$LOG" 2>&1 < /dev/null
