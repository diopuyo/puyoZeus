#!/bin/bash
# 真因A: 着地色をCNN==HSV観測一致で補正(--landing-observed-color, 42024d9) の検証。
# chainexitはdefault ON化済。比較: final(yellow->red=6470), hsvfb(B単独=5656,横滑り+1700)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
VIDEOS="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
HOLDOUT="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
VDIR="data/match_clips"; SI="0.03333333"
OUTDIR="data/verify/stable_cell_acc"; VIZDIR="data/verify/viz"; LOGDIR="logs/fix_v70_eval"
mkdir -p "${OUTDIR}" "${VIZDIR}" "${LOGDIR}"
BASE="--t2-highconf-yield --infer-empty-guard --no-constraint-fill"

run_eval(){ local name="$1"; shift
  PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
    --videos "${VIDEOS}" --holdout "${HOLDOUT}" --video-dir "${VDIR}" \
    --sample-interval "${SI}" --workers 8 \
    --output "${OUTDIR}/corruption_${name}_2026-06-01.json" ${BASE} "$@" \
    > "${LOGDIR}/eval_${name}_2026-06-01.log" 2>&1
  echo "[eval] ${name} 完了: $(date)" >> "${LOGDIR}/master_obscolor.log"; }

run_viz(){ local vid="$1"; local dir="$2"; shift 2
  PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
    --video "data/match_clips/${dir}/${vid}.mp4" \
    --output "${VIZDIR}/${vid}_obscolor_2026-06-01.mp4" ${BASE} "$@" \
    --dump-board-log-detailed "${VIZDIR}/${vid}_obscolor_2026-06-01.jsonl" \
    > "${LOGDIR}/viz_${vid}_obscolor_2026-06-01.log" 2>&1; }

echo "[start] obscolor: $(date)" > "${LOGDIR}/master_obscolor.log"
run_eval obscolor --landing-observed-color
run_eval obscolor_fb --landing-observed-color --hsv-classify-fallback
run_viz v89_match01 v89 --landing-observed-color --hsv-classify-fallback
run_viz v70_match01 v70 --landing-observed-color --hsv-classify-fallback
echo "[done] obscolor 全完了: $(date)" >> "${LOGDIR}/master_obscolor.log"
