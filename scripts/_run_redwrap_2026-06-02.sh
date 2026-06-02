#!/bin/bash
# 赤色相折り返し修正(--red-hue-wrap-fix, 3affa4f) の検証。chainexit default ON。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
VIDEOS="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
HOLDOUT="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
VDIR="data/match_clips"; SI="0.03333333"
OUTDIR="data/verify/stable_cell_acc"; VIZDIR="data/verify/viz"; LOGDIR="logs/fix_v70_eval"
mkdir -p "${OUTDIR}" "${VIZDIR}" "${LOGDIR}"
BASE="--t2-highconf-yield --infer-empty-guard --no-constraint-fill --red-hue-wrap-fix"
( PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
    --videos "${VIDEOS}" --holdout "${HOLDOUT}" --video-dir "${VDIR}" \
    --sample-interval "${SI}" --workers 6 \
    --output "${OUTDIR}/corruption_redwrap_2026-06-02.json" ${BASE} \
    > "${LOGDIR}/eval_redwrap_2026-06-02.log" 2>&1 ; echo "[eval] redwrap done $(date)" >> "${LOGDIR}/master_redwrap.log" ) &
E=$!
( for p in "v89_match01:v89" "v70_match01:v70"; do v="${p%%:*}"; d="${p##*:}"
    PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
      --video "data/match_clips/${d}/${v}.mp4" --output "${VIZDIR}/${v}_redwrap_2026-06-02.mp4" ${BASE} \
      --dump-board-log-detailed "${VIZDIR}/${v}_redwrap_2026-06-02.jsonl" > "${LOGDIR}/viz_${v}_redwrap.log" 2>&1
  done ; echo "[viz] redwrap done $(date)" >> "${LOGDIR}/master_redwrap.log" ) &
V=$!
echo "[start] redwrap $(date)" > "${LOGDIR}/master_redwrap.log"
wait $E $V
echo "[done] redwrap 全完了 $(date)" >> "${LOGDIR}/master_redwrap.log"
