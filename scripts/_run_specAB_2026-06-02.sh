#!/bin/bash
# 現コードでの specular-robust-saturation のクリーンA/B (Dのみ差分)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
V="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
H="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
OUT="data/verify/stable_cell_acc"; LOG="logs/fix_v70_eval"
BASE="--t2-highconf-yield --infer-empty-guard --no-constraint-fill"
run(){ local n="$1"; shift
 PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py --videos "$V" --holdout "$H" \
   --video-dir data/match_clips --sample-interval 0.03333333 --workers 8 \
   --output "$OUT/corruption_${n}_2026-06-02.json" $BASE "$@" > "$LOG/eval_${n}_2026-06-02.log" 2>&1
 echo "[eval] $n done $(date)" >> "$LOG/master_specAB.log"; }
echo "[start] specAB $(date)" > "$LOG/master_specAB.log"
run abNospec
run abSpec --specular-robust-saturation
echo "[done] specAB 全完了 $(date)" >> "$LOG/master_specAB.log"
