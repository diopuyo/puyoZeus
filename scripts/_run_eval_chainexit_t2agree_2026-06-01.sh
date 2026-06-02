#!/bin/bash
# 連鎖終了 game-event (940cf75) + T2 CNN/HSV合意解除 (77333a0) の eval + viz。
# 比較基準 = corruption_t2_guardB_nocfill_2026-06-01.json (2新機能なし)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}"

VIDEOS="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
HOLDOUT="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
VDIR="data/match_clips"
SI="0.03333333"
W="8"
OUTDIR="data/verify/stable_cell_acc"
VIZDIR="data/verify/viz"
LOGDIR="logs/fix_v70_eval"
mkdir -p "${OUTDIR}" "${VIZDIR}" "${LOGDIR}"

BASE="--t2-highconf-yield --infer-empty-guard --no-constraint-fill"

run_eval() {
  local name="$1"; shift
  echo "[eval] === ${name} 開始: $(date) ==="
  PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
    --videos "${VIDEOS}" --holdout "${HOLDOUT}" \
    --video-dir "${VDIR}" --sample-interval "${SI}" --workers "${W}" \
    --output "${OUTDIR}/corruption_${name}_2026-06-01.json" \
    ${BASE} "$@" > "${LOGDIR}/eval_${name}_2026-06-01.log" 2>&1
  echo "[eval] === ${name} 完了: $(date) ==="
}

run_viz() {
  local vid="$1"; local dir="$2"
  echo "[viz] === ${vid} 開始: $(date) ==="
  PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
    --video "data/match_clips/${dir}/${vid}.mp4" \
    --output "${VIZDIR}/${vid}_all_2026-06-01.mp4" \
    ${BASE} --game-event-chain-exit --t2-cnn-hsv-agree-yield \
    --dump-board-log-detailed "${VIZDIR}/${vid}_all_2026-06-01.jsonl" \
    > "${LOGDIR}/viz_${vid}_all_2026-06-01.log" 2>&1
  echo "[viz] === ${vid} 完了: $(date) ==="
}

run_eval chainexit --game-event-chain-exit
run_eval t2agree --t2-cnn-hsv-agree-yield
run_eval all --game-event-chain-exit --t2-cnn-hsv-agree-yield
run_viz v70_match01 v70
run_viz v89_match01 v89
echo "[done] chainexit/t2agree 全完了: $(date)"
