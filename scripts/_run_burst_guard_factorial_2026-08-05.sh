#!/bin/bash
# バーストガード factorial バックテスト再認識 (2026-08-05)。
# 引数1 = 構成名 (tguard / thr954 / full)、引数2 = MAXPAR (既定8)
# 4象限: OFF/0.97=初回走行(on_v2) / tguard=ON,0.97 / thr954=OFF,0.954 / full=ON,0.954
# 使い方: setsid -f bash scripts/_run_burst_guard_factorial_2026-08-05.sh full 8
set -u
CONFIG="${1:?構成名 (tguard/thr954/full) を指定}"
MAXPAR="${2:-8}"
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

JOBS="scripts/_jobs_burst_guard_${CONFIG}_2026-08-05.txt"
LOG="logs/burst_guard_${CONFIG}_2026-08-05.log"
mkdir -p "data/verify/burst_guard_2026-08-05/on_v2_${CONFIG}" logs

echo "[burst_guard_${CONFIG}] 開始 $(date) MAXPAR=${MAXPAR}" >> "${LOG}"
n=0
total=$(grep -c . "${JOBS}")
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  case "$cmd" in \#*) continue ;; esac
  while [ "$(jobs -r | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
  n=$((n+1))
  echo "[burst_guard_${CONFIG} $(date '+%H:%M:%S')] (${n}/${total}) start: ${cmd}" >> "${LOG}"
  bash -c "$cmd" >> "${LOG}" 2>&1 &
done < "${JOBS}"
wait
echo "[burst_guard_${CONFIG}] ALL DONE $(date) (${n} jobs)" >> "${LOG}"
