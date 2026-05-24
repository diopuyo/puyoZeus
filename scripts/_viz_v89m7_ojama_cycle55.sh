#!/bin/bash
# cycle 55 採否判定の ojama 確認用 viz 生成 (= eval 完了後に実行).
#
# v89m7 = ojama 降下 (= 相手連鎖からの送り) が複数回あるシーン。
# cycle 32 ojama 構造的除外 を撤回後、 既存 base model (= cnn_phase_b_large_v2.pt)
# の ojama 認識が cycle 55 fine-tune で壊れていないかをユーザー目視で確認する。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health cycle55_viz_v89m7_ojama

mkdir -p data/verify/cycle55_viz
mkdir -p logs/cycle55_viz

INPUT="data/phase_l/cut/v89m7_buf15s.mp4"
CNN_MODEL="models/cnn_cycle55.pt"
OUTPUT="data/verify/cycle55_viz/v89m7_ojama_cycle55.mp4"
BOARD_LOG="logs/cycle55_viz/v89m7_ojama_board.jsonl"

if [ ! -f "$INPUT" ]; then
  echo "[fail] input not found: $INPUT"
  finalize_health 1
  exit 1
fi
if [ ! -f "$CNN_MODEL" ]; then
  echo "[fail] cycle 55 model not found: $CNN_MODEL"
  finalize_health 1
  exit 1
fi

echo "=== v89m7 ojama viz (cycle 55) @ $(date) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
  --video "$INPUT" \
  --output "$OUTPUT" \
  --cnn-model "$CNN_MODEL" \
  --hsv-state data/per_video_hsv_ranges/_merged_default.json \
  --dump-board-log "$BOARD_LOG" \
  > logs/cycle55_viz/v89m7_ojama.log 2>&1

if [ ! -f "$OUTPUT" ]; then
  echo "[fail] viz output not generated"
  finalize_health 1
  exit 1
fi

echo "[done] viz -> $OUTPUT"
echo "[done] board_log -> $BOARD_LOG"

# 参考用に baseline (= cnn_phase_b_large_v2.pt) との比較 viz も並べて生成
BASELINE_OUTPUT="data/verify/cycle55_viz/v89m7_ojama_baseline.mp4"
BASELINE_LOG="logs/cycle55_viz/v89m7_ojama_baseline_board.jsonl"
if [ ! -f "$BASELINE_OUTPUT" ]; then
  echo "=== v89m7 ojama viz (baseline) @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$INPUT" \
    --output "$BASELINE_OUTPUT" \
    --cnn-model "models/cnn_phase_b_large_v2.pt" \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$BASELINE_LOG" \
    > logs/cycle55_viz/v89m7_ojama_baseline.log 2>&1
  echo "[done] baseline viz -> $BASELINE_OUTPUT"
fi

finalize_health 0
