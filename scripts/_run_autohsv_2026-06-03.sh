#!/bin/bash
# 汎用化測定: 自動HSVのみ(--no-per-video-hsv=全3軸自動)で16動画eval + 高画質viz。
# user目標「自動HSVだけで最低99.5%」の達成可否を測る。手調整あり99.87%との差=汎用化ギャップ。
# 全3軸自動ゆえCNN==HSV両方誤り合意のfail-silentリスク↑ → 盲点検知(check_three_way_sudden_drop)+viz目視で担保。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
V="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
H="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
VDIR="data/match_clips"; SI="0.03333333"
OUTDIR="data/verify/stable_cell_acc"; VIZDIR="data/verify/viz"; LOGDIR="logs/fix_v70_eval"
mkdir -p "${OUTDIR}" "${VIZDIR}" "${LOGDIR}"
MASTER="${LOGDIR}/master_autohsv.log"
echo "[start] autohsv eval $(date)" > "${MASTER}"
# 自動のみ eval (全3軸自動HSV)
PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
  --videos "${V}" --holdout "${H}" --video-dir "${VDIR}" --sample-interval "${SI}" --workers 6 \
  --no-per-video-hsv \
  --output "${OUTDIR}/auto_hsv_only_2026-06-03.json" \
  > "${LOGDIR}/eval_autohsv.log" 2>&1
echo "[eval] autohsv done $(date)" >> "${MASTER}"
# 自動のみ viz (高画質2本、user 21時目視用、board_logペア)
for p in "v89_match01:v89" "v29_match01:v29"; do v="${p%%:*}"; d="${p##*:}"
  PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
    --video "${VDIR}/${d}/${v}.mp4" --output "${VIZDIR}/${v}_autohsv_2026-06-03.mp4" \
    --no-per-video-hsv \
    --dump-board-log-detailed "${VIZDIR}/${v}_autohsv_2026-06-03.jsonl" \
    > "${LOGDIR}/viz_${v}_autohsv.log" 2>&1
done
echo "[done] autohsv 全完了 $(date)" >> "${MASTER}"
