#!/bin/bash
# 案B効果測定(c): 4条件フルON (enable_effect_gate + enable_effect_visual_gate) 再認識 (2026-08-04)。
#
# 構成 (memory project_session_2026-08-04_handoff 手順3):
#   (a) 全OFF   = 本番 npz (data/indicators_v2/boards_lean_regen_2026-07-31) 再利用
#   (b) 時間のみ = data/verify/effect_gate_2026-08-03_v2/on 再利用 (batch1分)
#   (c) 4条件フルON = 本スクリプトで新規再認識 (batch1+batch2 の ok/fixed ラベル
#       保有 31 動画、ラベル最大 t_sec + 8 秒まで)
#
# pipeline 設定はラベル元 (_jobs_lean_regen_2026-07-31.txt) と同一
# (--enable-chain-tracker --with-next) + ゲート2フラグのみ追加。
# 突合は measure_effect_gate_impact のアンカー frame_idx bit一致方式。
#
# 使い方: setsid -f bash scripts/_run_effect_gate_c_2026-08-04.sh [MAXPAR=8]
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

JOBS="scripts/_jobs_effect_gate_c_2026-08-04.txt"
MAXPAR="${1:-8}"
LOG="logs/effect_gate_c_2026-08-04.log"
mkdir -p data/verify/effect_gate_2026-08-04_c/on_full logs

echo "[effect_gate_c] 開始 $(date) MAXPAR=${MAXPAR}" >> "${LOG}"
n=0
total=$(grep -c . "${JOBS}")
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  case "$cmd" in \#*) continue ;; esac
  while [ "$(jobs -r | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
  n=$((n+1))
  echo "[effect_gate_c $(date '+%H:%M:%S')] (${n}/${total}) start: ${cmd}" >> "${LOG}"
  bash -c "$cmd" >> "${LOG}" 2>&1 &
done < "${JOBS}"
wait
echo "[effect_gate_c] ALL DONE $(date) (${n} jobs)" >> "${LOG}"
