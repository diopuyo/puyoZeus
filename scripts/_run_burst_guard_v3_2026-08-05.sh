#!/bin/bash
# バーストガードv3 (Stage1.5+1.5b+§12close拡張+閾値0.954) バックテスト再認識 (2026-08-05)。
# 使い方: setsid -f bash scripts/_run_burst_guard_v3_2026-08-05.sh [MAXPAR=8]
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

JOBS="scripts/_jobs_burst_guard_v4_2026-08-05.txt"
MAXPAR="${1:-8}"
LOG="logs/burst_guard_v3_2026-08-05.log"
mkdir -p data/verify/burst_guard_2026-08-05/on_v4 logs

echo "[burst_guard_v3] 開始 $(date) MAXPAR=${MAXPAR}" >> "${LOG}"
n=0
total=$(grep -c . "${JOBS}")
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  case "$cmd" in \#*) continue ;; esac
  while [ "$(jobs -r | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
  n=$((n+1))
  echo "[burst_guard_v3 $(date '+%H:%M:%S')] (${n}/${total}) start: ${cmd}" >> "${LOG}"
  bash -c "$cmd" >> "${LOG}" 2>&1 &
done < "${JOBS}"
wait
echo "[burst_guard_v3] ALL DONE $(date) (${n} jobs)" >> "${LOG}"
