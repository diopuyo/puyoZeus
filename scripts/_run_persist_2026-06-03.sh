#!/bin/bash
# (a) 持続corruption指標で16動画eval。真の持続精度(1fr点滅除外)を測定。
# default(採用スタック=機能D ON含む)。persist閾値はライブラリ既定(CORRUPTION_PERSIST_MIN_FRAMES=3)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
V="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
H="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
OUTDIR="data/verify/stable_cell_acc"; LOGDIR="logs/fix_v70_eval"
mkdir -p "${OUTDIR}" "${LOGDIR}"
echo "[start] persist eval $(date)" > "${LOGDIR}/master_persist.log"
PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
  --videos "${V}" --holdout "${H}" --video-dir data/match_clips --sample-interval 0.03333333 --workers 6 \
  --output "${OUTDIR}/corruption_persist_2026-06-03.json" \
  > "${LOGDIR}/eval_persist.log" 2>&1
echo "[done] persist eval 全完了 $(date)" >> "${LOGDIR}/master_persist.log"
