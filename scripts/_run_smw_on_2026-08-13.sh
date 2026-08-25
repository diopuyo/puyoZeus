#!/bin/bash
# stable_majority_window ON 収集ジョブ実行 (2026-08-13、一時検証用)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

JOBS="scripts/_jobs_smw_on_2026-08-13.txt"
MAXPAR=4
LOG="logs/smw_on_2026-08-13.log"
mkdir -p data/verify/board_labels_smw_on_2026-08-13 logs
: > "${LOG}"

echo "[smw_on] 開始 $(date)" >> "${LOG}"
n=0
total=$(grep -c . "${JOBS}")
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  while [ "$(jobs -r | wc -l)" -ge "$MAXPAR" ]; do sleep 3; done
  n=$((n+1))
  echo "[smw_on $(date '+%H:%M:%S')] (${n}/${total}) start: ${cmd}" >> "${LOG}"
  bash -c "$cmd" >> "${LOG}" 2>&1 &
done < "${JOBS}"
wait
echo "[smw_on] ALL DONE $(date) (${n} jobs)" >> "${LOG}"
