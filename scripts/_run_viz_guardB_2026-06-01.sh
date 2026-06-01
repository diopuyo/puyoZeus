#!/bin/bash
# guardB 案B (+t2yield+nocfill = 最終候補) の viz。ユーザー目視レビュー用。
# 2026-05-31 combined (t2yield+nocfill、guardなし) との差分で col0 FP が消えるか / 新たにぷよを落とさないかを確認。
# board_log ペア保存 (feedback_viz_boardlog_pairing)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}"
OUT="data/verify/viz"
mkdir -p "${OUT}"

run_viz() {
  local vid="$1"; local dir="$2"
  echo "[viz] === ${vid} 開始: $(date) ==="
  PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
    --video "data/match_clips/${dir}/${vid}.mp4" \
    --output "${OUT}/${vid}_t2_guardB_nocfill_2026-06-01.mp4" \
    --no-constraint-fill --t2-highconf-yield --infer-empty-guard \
    --dump-board-log-detailed "${OUT}/${vid}_t2_guardB_nocfill_2026-06-01.jsonl" \
    > "logs/fix_v70_eval/viz_${vid}_guardB_2026-06-01.log" 2>&1
  echo "[viz] === ${vid} 完了: $(date) ==="
}

run_viz v70_match01 v70
run_viz v89_match01 v89
echo "[viz] guardB viz 全完了: $(date)"
