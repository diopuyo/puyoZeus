#!/bin/bash
# 自分の監督プロセスだけを一時停止し、完了済みqの監査を先行する。
set -euo pipefail
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
mapfile -t SUPERVISORS < <(pgrep -f '^/mnt/.*/venv/bin/python -B -m scripts.prefire_v6_batch$')
(( ${#SUPERVISORS[@]} == 1 )) || exit 2
PARENT_PID=${SUPERVISORS[0]}
AVAILABLE=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
COUNT=$(pgrep -fc '^/mnt/.*/venv/bin/python' || true)
(( AVAILABLE >= 6291456 && COUNT < 8 )) || exit 3
kill -STOP "$PARENT_PID"
trap 'kill -CONT "$PARENT_PID"' EXIT
echo "監督のみ一時停止: $PARENT_PID。既存再生の子プロセスは継続。"
"$PY" -B -c 'from scripts.prefire_v6_batch import cuts; cuts()'
echo '打切り再生完了、監督を再開する。'
