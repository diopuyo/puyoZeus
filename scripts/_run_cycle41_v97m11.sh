#!/bin/bash
# cycle 41: PuyoPresenceGate 有効化 (= cycle 32e で実装済、 use_puyo_gate=True)
# 既 default model (= cnn_phase_b_large_v2.pt) + gate で 「puyo らしさ」 事前判定
# 時間節約のため v97m11 1 本で先に判定、 効果あれば 3 動画展開
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/review_videos/cycle41
mkdir -p logs/board_logs

PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
  --video data/evaluation_videos/v97_match11_96s.mp4 \
  --output data/review_videos/cycle41/cycle41_v97m11.mp4 \
  --cnn-model models/cnn_phase_b_large_v2.pt \
  --hsv-state data/per_video_hsv_ranges/_merged_default.json \
  --dump-board-log logs/board_logs/cycle41_v97m11.jsonl \
  --use-puyo-gate \
  > logs/cycle_41_viz_v97m11.log 2>&1

mkdir -p data/verify/cycle41_eval
PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
  --board-log logs/board_logs/cycle41_v97m11.jsonl \
  --report-out data/verify/cycle41_eval/cycle41_v97m11.json \
  > logs/cycle_41_eval_v97m11.log 2>&1

echo "=== cycle 41 v97m11 DONE @ $(date) ===" | tee logs/cycle_41_v97m11_done.flag
