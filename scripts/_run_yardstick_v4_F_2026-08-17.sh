#!/bin/bash
# 一般分布退行チェック: 物差し52盤面の構成F (本番採用13フラグ) 再認識 (2026-08-17)
# 使い方: setsid -f bash scripts/_run_yardstick_v4_F_2026-08-17.sh [MAXPAR=4]
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

JOBS="scripts/_jobs_yardstick_v4_F_2026-08-17.txt"
MAXPAR="${1:-4}"
LOG="logs/yardstick_v4_F_2026-08-17.log"
mkdir -p data/verify/board_labels_v4F_yardstick_2026-08-17 logs

echo "[yardstick_v4_F] 開始 $(date) MAXPAR=${MAXPAR}" >> "${LOG}"
n=0
total=$(grep -c . "${JOBS}")
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  case "$cmd" in \#*) continue ;; esac
  while [ "$(jobs -r | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
  n=$((n+1))
  echo "[yardstick_v4_F $(date '+%H:%M:%S')] (${n}/${total}) start: ${cmd}" >> "${LOG}"
  bash -c "$cmd" >> "${LOG}" 2>&1 &
done < "${JOBS}"
wait
echo "[yardstick_v4_F] ALL DONE $(date) (${n} jobs)" >> "${LOG}"
