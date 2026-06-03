#!/bin/bash
# 予告発光ガード v2(ターゲット型=おじゃま誤認セルのみ復元)A/B。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
V="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
H="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
VDIR="data/match_clips"; SI="0.03333333"; OUTDIR="data/verify/stable_cell_acc"; VIZDIR="data/verify/viz"; LOGDIR="logs/fix_v70_eval"
MASTER="${LOGDIR}/master_glow_v2.log"; echo "[start] glow v2 A/B $(date)" > "${MASTER}"
PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
  --videos "${V}" --holdout "${H}" --video-dir "${VDIR}" --sample-interval "${SI}" --workers 6 \
  --ojama-warning-glow-guard \
  --output "${OUTDIR}/corruption_glow_v2_2026-06-04.json" > "${LOGDIR}/eval_glow_v2.log" 2>&1
echo "[eval] glow v2 done $(date)" >> "${MASTER}"
PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
  --video "${VDIR}/v89/v89_match01.mp4" --output "${VIZDIR}/v89_match01_glowV2_2026-06-04.mp4" --ojama-warning-glow-guard \
  --dump-board-log-detailed "${VIZDIR}/v89_match01_glowV2_2026-06-04.jsonl" > "${LOGDIR}/viz_v89_glowV2.log" 2>&1
echo "[done] glow v2 全完了 $(date)" >> "${MASTER}"
