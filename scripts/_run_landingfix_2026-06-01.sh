#!/bin/bash
# 着地色修正 案1 (d6576ab、--landing-color-fix) の eval + viz を並列実行。
# 比較基準 = corruption_chainexit_2026-06-01.json (landing-color-fix なし、他同条件)。
# viz は診断フィールド(falling_pair old/new)込みで仮説裏取り + 着地直後誤認の目視。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}"

VIDEOS="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
HOLDOUT="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
VDIR="data/match_clips"
SI="0.03333333"
OUTDIR="data/verify/stable_cell_acc"
VIZDIR="data/verify/viz"
LOGDIR="logs/fix_v70_eval"
mkdir -p "${OUTDIR}" "${VIZDIR}" "${LOGDIR}"
BASE="--t2-highconf-yield --infer-empty-guard --no-constraint-fill --game-event-chain-exit"

# eval (workers6: viz と並列のため VRAM 余裕を残す)
( PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
    --videos "${VIDEOS}" --holdout "${HOLDOUT}" \
    --video-dir "${VDIR}" --sample-interval "${SI}" --workers 6 \
    --output "${OUTDIR}/corruption_landingfix_2026-06-01.json" \
    ${BASE} --landing-color-fix > "${LOGDIR}/eval_landingfix_2026-06-01.log" 2>&1 ; \
  echo "[eval] landingfix 完了: $(date)" >> "${LOGDIR}/master_landingfix.log" ) &
EVAL_PID=$!

# viz (診断フィールド込み、v89/v70)
( for pair in "v89_match01:v89" "v70_match01:v70"; do
    vid="${pair%%:*}"; dir="${pair##*:}"
    PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
      --video "data/match_clips/${dir}/${vid}.mp4" \
      --output "${VIZDIR}/${vid}_landingfix_2026-06-01.mp4" \
      ${BASE} --landing-color-fix \
      --dump-board-log-detailed "${VIZDIR}/${vid}_landingfix_2026-06-01.jsonl" \
      > "${LOGDIR}/viz_${vid}_landingfix_2026-06-01.log" 2>&1
  done ; echo "[viz] landingfix 完了: $(date)" >> "${LOGDIR}/master_landingfix.log" ) &
VIZ_PID=$!

echo "[start] eval(pid=$EVAL_PID) + viz(pid=$VIZ_PID) 並列起動: $(date)" > "${LOGDIR}/master_landingfix.log"
wait $EVAL_PID $VIZ_PID
echo "[done] landingfix eval+viz 全完了: $(date)" >> "${LOGDIR}/master_landingfix.log"
