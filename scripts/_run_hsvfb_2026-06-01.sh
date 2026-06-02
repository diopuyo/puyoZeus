#!/bin/bash
# HSV分類fallback(2ca5e33) + eval信頼性指標(bf961aa) の検証。
# stack = t2yield+guard+nocfill+chainexit①のみ + hsv-classify-fallback (landing-color-fixは不発のため除外)。
# 比較: final(yellow->red=6470)。新指標 confirmed_majority_agree_rate / avg_puyo per-side / non_stable chain除外 も出力。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}"
VIDEOS="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
HOLDOUT="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
VDIR="data/match_clips"; SI="0.03333333"
OUTDIR="data/verify/stable_cell_acc"; VIZDIR="data/verify/viz"; LOGDIR="logs/fix_v70_eval"
mkdir -p "${OUTDIR}" "${VIZDIR}" "${LOGDIR}"
BASE="--t2-highconf-yield --infer-empty-guard --no-constraint-fill --game-event-chain-exit --hsv-classify-fallback"

( PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
    --videos "${VIDEOS}" --holdout "${HOLDOUT}" --video-dir "${VDIR}" \
    --sample-interval "${SI}" --workers 6 \
    --output "${OUTDIR}/corruption_hsvfb_2026-06-01.json" ${BASE} \
    > "${LOGDIR}/eval_hsvfb_2026-06-01.log" 2>&1 ; \
  echo "[eval] hsvfb 完了: $(date)" >> "${LOGDIR}/master_hsvfb.log" ) &
EVAL_PID=$!
( for pair in "v89_match01:v89" "v70_match01:v70"; do
    vid="${pair%%:*}"; dir="${pair##*:}"
    PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
      --video "data/match_clips/${dir}/${vid}.mp4" \
      --output "${VIZDIR}/${vid}_hsvfb_2026-06-01.mp4" ${BASE} \
      --dump-board-log-detailed "${VIZDIR}/${vid}_hsvfb_2026-06-01.jsonl" \
      > "${LOGDIR}/viz_${vid}_hsvfb_2026-06-01.log" 2>&1
  done ; echo "[viz] hsvfb 完了: $(date)" >> "${LOGDIR}/master_hsvfb.log" ) &
VIZ_PID=$!
echo "[start] hsvfb eval+viz 並列: $(date)" > "${LOGDIR}/master_hsvfb.log"
wait $EVAL_PID $VIZ_PID
echo "[done] hsvfb eval+viz 全完了: $(date)" >> "${LOGDIR}/master_hsvfb.log"
