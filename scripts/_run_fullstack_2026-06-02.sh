#!/bin/bash
# 全候補スタック総合検証: 連鎖終了(default)+赤折返し(default)+D+フェーズA状態認識+双方向ゲート。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
V="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
H="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
OUT="data/verify/stable_cell_acc"; VIZ="data/verify/viz"; LOG="logs/fix_v70_eval"
FLAGS="--t2-highconf-yield --infer-empty-guard --no-constraint-fill --enable-ojama-visual-detection --ojama-tier1-warmup --specular-robust-saturation --stable-recovery-gate"
( PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py --videos "$V" --holdout "$H" \
   --video-dir data/match_clips --sample-interval 0.03333333 --workers 6 \
   --output "$OUT/corruption_fullstack_2026-06-02.json" $FLAGS \
   > "$LOG/eval_fullstack_2026-06-02.log" 2>&1; echo "[eval] fullstack done $(date)" >> "$LOG/master_fullstack.log" ) &
E=$!
( PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py --video data/match_clips/v70/v70_match01.mp4 \
   --output "$VIZ/v70_match01_full_2026-06-02.mp4" $FLAGS \
   --dump-board-log-detailed "$VIZ/v70_match01_full_2026-06-02.jsonl" > "$LOG/viz_v70_full.log" 2>&1
  echo "[viz] v70 done $(date)" >> "$LOG/master_fullstack.log" ) &
Vp=$!
echo "[start] fullstack $(date)" > "$LOG/master_fullstack.log"
wait $E $Vp
echo "[done] fullstack 全完了 $(date)" >> "$LOG/master_fullstack.log"
