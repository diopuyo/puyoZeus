#!/bin/bash
# 案X (chain_exit_next_signal) A/B。baseline=現default(P3 OFF)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
V="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
H="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
VDIR="data/match_clips"; SI="0.03333333"; OUT="data/verify/stable_cell_acc"; VIZ="data/verify/viz"; LOG="logs/fix_v70_eval"
M="${LOG}/master_caseX.log"; echo "[start] caseX A/B $(date)" > "$M"
# caseX ON eval (baseは既存 corruption_p3base_2026-06-05.json=現default を流用)
PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py --videos "$V" --holdout "$H" --video-dir "$VDIR" --sample-interval "$SI" --workers 6 \
  --chain-exit-next-signal --output "$OUT/corruption_caseX_2026-06-05.json" > "$LOG/eval_caseX.log" 2>&1
echo "[eval] caseX ON done $(date)" >> "$M"
# viz: v89_match01, v70_match02 を caseX ON(board_logペア)
for p in "v89_match01:v89" "v70_match02:v70"; do v="${p%%:*}"; d="${p##*:}"
  PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py --video "$VDIR/$d/$v.mp4" --output "$VIZ/${v}_caseX_2026-06-05.mp4" --chain-exit-next-signal \
    --dump-board-log-detailed "$VIZ/${v}_caseX_2026-06-05.jsonl" > "$LOG/viz_${v}_caseX.log" 2>&1
done
echo "[done] caseX 全完了 $(date)" >> "$M"
