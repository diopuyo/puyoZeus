#!/bin/bash
# 案X + warmup0.5s 再eval(退行救済確認)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
V="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
H="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
OUT="data/verify/stable_cell_acc"; LOG="logs/fix_v70_eval"; M="${LOG}/master_caseXw.log"
echo "[start] caseXw $(date)" > "$M"
PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py --videos "$V" --holdout "$H" --video-dir data/match_clips --sample-interval 0.03333333 --workers 6 \
  --chain-exit-next-signal --output "$OUT/corruption_caseXw_2026-06-05.json" > "$LOG/eval_caseXw.log" 2>&1
echo "[done] caseXw 全完了 $(date)" >> "$M"
