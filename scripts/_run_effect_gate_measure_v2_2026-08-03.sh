#!/bin/bash
# エフェクト時間ゲート効果測定 v2 (2026-08-03): --with-next 抜け修正版。
#
# v1 (_run_effect_gate_measure_2026-08-03.sh) はラベル元 npz
# (data/indicators_v2/boards_lean_regen_2026-07-31) を生成した
# _jobs_lean_regen_2026-07-31.txt と異なる pipeline 設定 (--with-next 抜け)
# で再収集しており、NextDetector 不在で着地色推論経路が変わるため
# 「ok」ラベル (誤り0のはず) でも非ゼロ誤りが出る事故が発生した (検収不能)。
# v2 は --with-next を追加し、ラベル元と同一設定で再現性を確保する。
#
# 使い方: setsid -f bash scripts/_run_effect_gate_measure_v2_2026-08-03.sh [MAXPAR=8]
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

JOBS="scripts/_jobs_effect_gate_measure_v2_2026-08-03.txt"
MAXPAR="${1:-8}"
LOG="logs/effect_gate_measure_v2_2026-08-03.log"
mkdir -p data/verify/effect_gate_2026-08-03_v2/off data/verify/effect_gate_2026-08-03_v2/on logs

echo "[effect_gate_measure_v2] 開始 $(date) MAXPAR=${MAXPAR}" >> "${LOG}"
n=0
total=$(grep -c . "${JOBS}")
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  case "$cmd" in \#*) continue ;; esac
  while [ "$(jobs -r | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
  n=$((n+1))
  echo "[effect_gate_measure_v2 $(date '+%H:%M:%S')] (${n}/${total}) start: ${cmd}" >> "${LOG}"
  bash -c "$cmd" >> "${LOG}" 2>&1 &
done < "${JOBS}"
wait
echo "[effect_gate_measure_v2] ALL DONE $(date) (${n} jobs)" >> "${LOG}"
