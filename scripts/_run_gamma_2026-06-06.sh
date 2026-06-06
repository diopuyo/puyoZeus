#!/bin/bash
# 案γ(slide_override_ojama_hold)A/B。gsettle(default ON)+案γ。baseline比較=corruption_gsettle_2026-06-06.json。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
V="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
H="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
OUT="data/verify/stable_cell_acc"; VIZ="data/verify/viz"; LOG="logs/fix_v70_eval"; M="${LOG}/master_gamma.log"
echo "[start] gamma $(date)" > "$M"
PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py --videos "$V" --holdout "$H" --video-dir data/match_clips --sample-interval 0.03333333 --workers 6 \
  --gravity-settle-state --slide-override-ojama-hold --output "$OUT/corruption_gamma_2026-06-06.json" > "$LOG/eval_gamma.log" 2>&1
echo "[eval] gamma done $(date)" >> "$M"
for p in "v89_match01:v89" "v70_match02:v70"; do v="${p%%:*}"; d="${p##*:}"
  PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py --video "data/match_clips/$d/$v.mp4" --output "$VIZ/${v}_gamma_2026-06-06.mp4" --gravity-settle-state --slide-override-ojama-hold \
    --dump-board-log-detailed "$VIZ/${v}_gamma_2026-06-06.jsonl" > "$LOG/viz_${v}_gamma.log" 2>&1
done
echo "[done] gamma 全完了 $(date)" >> "$M"
