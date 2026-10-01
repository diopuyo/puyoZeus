#!/bin/bash
# 当worktree専用。全WSL検証Pythonを数え、起動時にメモリ余裕を確認する。
set -euo pipefail
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
AVAILABLE=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
(( AVAILABLE >= 5242880 )) || { echo 'available < 5GiB (4GiB + reserve)'; exit 3; }
COUNT=$(pgrep -fc '[v]env/bin/python.*(-m scripts\.|-m pytest)' || true)
(( COUNT < 8 )) || { echo '8 processes already running'; exit 4; }
mkdir -p logs/prefire_prediction/v5c
JOB=$1
shift
LOG=logs/prefire_prediction/v5c/$JOB.log
[[ ! -e "$LOG" ]] || { echo 'existing log'; exit 5; }
printf '%s available_kib=%s workers_before=%s job=%s\n' "$(date -Iseconds)" "$AVAILABLE" "$COUNT" "$JOB" >> logs/prefire_prediction/v5c/resources.log
setsid -f bash -c 'exec nice -n 10 "$0" -B "$@"' "$PY" "$@" > "$LOG" 2>&1 < /dev/null
