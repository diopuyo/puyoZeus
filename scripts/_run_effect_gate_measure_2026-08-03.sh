#!/bin/bash
# エフェクト時間ゲート (enable_effect_gate) 効果測定: 満杯盤面ラベル24動画分の
# OFF/ON board npz を並列収集する (2026-08-03)。
# 熱対策は解除済み (feedback_thermal_safety_mandatory) のため cooldown なし、
# 並列数は実績並列 (collect_indicators_v2 で14並列実績) より控えめの MAXPAR に設定。
#
# 使い方: setsid -f bash scripts/_run_effect_gate_measure_2026-08-03.sh [MAXPAR=8]
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

JOBS="scripts/_jobs_effect_gate_measure_2026-08-03.txt"
MAXPAR="${1:-8}"
LOG="logs/effect_gate_measure_2026-08-03.log"
mkdir -p data/verify/effect_gate_2026-08-03/off data/verify/effect_gate_2026-08-03/on logs

echo "[effect_gate_measure] 開始 $(date) MAXPAR=${MAXPAR}" >> "${LOG}"
n=0
total=$(grep -c . "${JOBS}")
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  case "$cmd" in \#*) continue ;; esac
  while [ "$(jobs -r | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
  n=$((n+1))
  echo "[effect_gate_measure $(date '+%H:%M:%S')] (${n}/${total}) start: ${cmd}" >> "${LOG}"
  bash -c "$cmd" >> "${LOG}" 2>&1 &
done < "${JOBS}"
wait
echo "[effect_gate_measure] ALL DONE $(date) (${n} jobs)" >> "${LOG}"
