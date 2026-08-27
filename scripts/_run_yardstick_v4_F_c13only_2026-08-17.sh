#!/bin/bash
# 一般分布退行チェック: c13サブセットのみ (video再DL drift無し、W8確認済み) の構成F収集 (2026-08-17)
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

JOBS="scripts/_jobs_yardstick_v4_F_c13only_2026-08-17.txt"
LOG="logs/yardstick_v4_F_c13only_2026-08-17.log"
mkdir -p data/verify/board_labels_v4F_yardstick_2026-08-17 logs

echo "[c13only] 開始 $(date)" > "${LOG}"
n=0
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  n=$((n+1))
  echo "[c13only $(date '+%H:%M:%S')] (${n}) start: ${cmd}" >> "${LOG}"
  bash -c "$cmd" >> "${LOG}" 2>&1 &
done < "${JOBS}"
wait
echo "[c13only] ALL DONE $(date) (${n} jobs)" >> "${LOG}"
