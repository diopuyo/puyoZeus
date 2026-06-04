#!/bin/bash
# 長時間自律: glow_v4完了待ち → 不具合A(chain-refire-cooldown)A/B → 広域汎用化テスト(39 phase_l動画 auto-HSV)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
OUTDIR="data/verify/stable_cell_acc"; LOGDIR="logs/fix_v70_eval"; VDIR="data/match_clips"; SI="0.03333333"
MASTER="${LOGDIR}/master_overnight.log"; echo "[overnight start] $(date)" > "${MASTER}"
V16="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
H16="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"

# 0. glow_v4 完了待ち(最大40分)
for i in $(seq 1 80); do
  [ -f "${OUTDIR}/corruption_glow_v4_2026-06-04.json" ] && { echo "[overnight] glow_v4 done $(date)" >> "${MASTER}"; break; }
  sleep 30
done

# 1. 不具合A: chain-refire-cooldown A/B (16動画)
PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
  --videos "${V16}" --holdout "${H16}" --video-dir "${VDIR}" --sample-interval "${SI}" --workers 6 \
  --chain-refire-cooldown \
  --output "${OUTDIR}/corruption_chaincooldown_2026-06-04.json" > "${LOGDIR}/eval_chaincooldown.log" 2>&1
echo "[overnight] chain-cooldown A/B done $(date)" >> "${MASTER}"
# v89 chain境界 board_log (OFF/ON比較用、ON)
PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
  --video "${VDIR}/v89/v89_match01.mp4" --output "${OUTDIR}/../viz/v89_chaincooldown_2026-06-04.mp4" --chain-refire-cooldown \
  --dump-board-log-detailed "${OUTDIR}/../viz/v89_chaincooldown_2026-06-04.jsonl" > "${LOGDIR}/viz_v89_chaincooldown.log" 2>&1
echo "[overnight] chain-cooldown viz done $(date)" >> "${MASTER}"

# 2. 広域汎用化テスト: phase_l/cut の全クリップを auto-HSV(--no-per-video-hsv) で eval
GVIDS=$(ls data/phase_l/cut/*.mp4 2>/dev/null | xargs -n1 basename | sed 's/.mp4$//' | paste -sd, -)
echo "[overnight] 汎用化テスト対象: $(echo $GVIDS | tr ',' '\n' | wc -l)動画" >> "${MASTER}"
PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
  --videos "${GVIDS}" --holdout "" --video-dir data/phase_l/cut --sample-interval 0.06666666 --workers 6 \
  --no-per-video-hsv \
  --output "${OUTDIR}/generalization_phaseL_autohsv_2026-06-04.json" > "${LOGDIR}/eval_generalization.log" 2>&1
echo "[overnight done] 全完了 $(date)" >> "${MASTER}"
