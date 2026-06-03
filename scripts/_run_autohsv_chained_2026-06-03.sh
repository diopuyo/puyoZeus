#!/bin/bash
# (a)persist eval 完了を待ってから autohsv eval+viz を実行(競合回避の連鎖)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
LOGDIR="logs/fix_v70_eval"; OUTDIR="data/verify/stable_cell_acc"; VIZDIR="data/verify/viz"; VDIR="data/match_clips"
MASTER="${LOGDIR}/master_autohsv_chain.log"
echo "[chain start] $(date)" > "${MASTER}"
# 1. persist eval 完了待ち (最大90分)
for i in $(seq 1 180); do
  if [ -f "${OUTDIR}/corruption_persist_2026-06-03.json" ]; then echo "[chain] persist done $(date)" >> "${MASTER}"; break; fi
  sleep 30
done
V="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
H="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
# 2. autohsv eval (自動のみ=--no-per-video-hsv, store_true修正済)
PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
  --videos "${V}" --holdout "${H}" --video-dir "${VDIR}" --sample-interval 0.03333333 --workers 6 \
  --no-per-video-hsv \
  --output "${OUTDIR}/auto_hsv_only_2026-06-03.json" \
  > "${LOGDIR}/eval_autohsv.log" 2>&1
echo "[chain] autohsv eval done $(date)" >> "${MASTER}"
# 3. autohsv viz (高画質2本, 21時目視用)
for p in "v89_match01:v89" "v29_match01:v29"; do v="${p%%:*}"; d="${p##*:}"
  PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
    --video "${VDIR}/${d}/${v}.mp4" --output "${VIZDIR}/${v}_autohsv_2026-06-03.mp4" --no-per-video-hsv \
    --dump-board-log-detailed "${VIZDIR}/${v}_autohsv_2026-06-03.jsonl" \
    > "${LOGDIR}/viz_${v}_autohsv.log" 2>&1
done
echo "[chain done] autohsv 全完了 $(date)" >> "${MASTER}"
