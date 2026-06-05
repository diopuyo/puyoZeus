#!/bin/bash
# 不具合A 案P3(chain_max_hold_override)A/B。baseline=現default(glow ON含む)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
V="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
H="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
VDIR="data/match_clips"; SI="0.03333333"; OUT="data/verify/stable_cell_acc"; VIZ="data/verify/viz"; LOG="logs/fix_v70_eval"
M="${LOG}/master_p3.log"; echo "[start] P3 A/B $(date)" > "$M"
# baseline (現default, P3 OFF)
PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py --videos "$V" --holdout "$H" --video-dir "$VDIR" --sample-interval "$SI" --workers 6 \
  --output "$OUT/corruption_p3base_2026-06-05.json" > "$LOG/eval_p3base.log" 2>&1
echo "[eval] P3 baseline done $(date)" >> "$M"
# P3 ON
PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py --videos "$V" --holdout "$H" --video-dir "$VDIR" --sample-interval "$SI" --workers 6 \
  --chain-max-hold-override --output "$OUT/corruption_p3on_2026-06-05.json" > "$LOG/eval_p3on.log" 2>&1
echo "[eval] P3 ON done $(date)" >> "$M"
# viz: v89_match01, v70_match02 を P3 ON(board_logペア)
for p in "v89_match01:v89" "v70_match02:v70"; do v="${p%%:*}"; d="${p##*:}"
  PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py --video "$VDIR/$d/$v.mp4" --output "$VIZ/${v}_p3on_2026-06-05.mp4" --chain-max-hold-override \
    --dump-board-log-detailed "$VIZ/${v}_p3on_2026-06-05.jsonl" > "$LOG/viz_${v}_p3.log" 2>&1
done
echo "[done] P3 全完了 $(date)" >> "$M"
