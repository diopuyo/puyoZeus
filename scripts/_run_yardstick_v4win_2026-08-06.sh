#!/bin/bash
# 99.99%再測定 (正式版): 物差し14動画の全編v4再認識 (2026-08-05)
# 窓付き収集は色較正コールドスタートで物差し前提を満たさず棄却 → 全編収集に切替
# 使い方: setsid -f bash scripts/_run_yardstick_v4win_2026-08-05.sh [MAXPAR=8]
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

JOBS="scripts/_jobs_yardstick_v4win_2026-08-06.txt"
MAXPAR="${1:-8}"
LOG="logs/yardstick_v4win_2026-08-05.log"
mkdir -p data/verify/board_labels_v4_win_2026-08-06 logs

echo "[yardstick_v4win] 開始 $(date) MAXPAR=${MAXPAR}" >> "${LOG}"
n=0
total=$(grep -c . "${JOBS}")
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  case "$cmd" in \#*) continue ;; esac
  while [ "$(jobs -r | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
  n=$((n+1))
  echo "[yardstick_v4win $(date '+%H:%M:%S')] (${n}/${total}) start: ${cmd}" >> "${LOG}"
  bash -c "$cmd" >> "${LOG}" 2>&1 &
done < "${JOBS}"
wait
echo "[yardstick_v4win] ALL DONE $(date) (${n} jobs)" >> "${LOG}"
