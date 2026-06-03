#!/bin/bash
# 予告発光ガード(--ojama-warning-glow-guard)のA/B検証。
# eval: glow ON 16動画 (baselineは既存 default=glow OFF と比較)。
# viz: v89_match01 を glow OFF/ON ペアで生成(t≈70 黄→O誤認が消えるか目視)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
V="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
H="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
VDIR="data/match_clips"; SI="0.03333333"; OUTDIR="data/verify/stable_cell_acc"; VIZDIR="data/verify/viz"; LOGDIR="logs/fix_v70_eval"
mkdir -p "${OUTDIR}" "${VIZDIR}" "${LOGDIR}"; MASTER="${LOGDIR}/master_glow.log"
echo "[start] glow A/B $(date)" > "${MASTER}"
# eval glow ON
PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
  --videos "${V}" --holdout "${H}" --video-dir "${VDIR}" --sample-interval "${SI}" --workers 6 \
  --ojama-warning-glow-guard \
  --output "${OUTDIR}/corruption_glow_2026-06-04.json" > "${LOGDIR}/eval_glow.log" 2>&1
echo "[eval] glow done $(date)" >> "${MASTER}"
# viz v89_match01: glow OFF と ON (board_logペア)
PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
  --video "${VDIR}/v89/v89_match01.mp4" --output "${VIZDIR}/v89_match01_glowOFF_2026-06-04.mp4" \
  --dump-board-log-detailed "${VIZDIR}/v89_match01_glowOFF_2026-06-04.jsonl" > "${LOGDIR}/viz_v89_glowOFF.log" 2>&1
PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
  --video "${VDIR}/v89/v89_match01.mp4" --output "${VIZDIR}/v89_match01_glowON_2026-06-04.mp4" --ojama-warning-glow-guard \
  --dump-board-log-detailed "${VIZDIR}/v89_match01_glowON_2026-06-04.jsonl" > "${LOGDIR}/viz_v89_glowON.log" 2>&1
echo "[done] glow A/B 全完了 $(date)" >> "${MASTER}"
