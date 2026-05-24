#!/bin/bash
# cycle 40: cnn_override_prob 0.70 → 0.90 で HSV 主軸化試行 (= v89m3 のみで時間節約)
# 先行研究 puyogg がシンプル MLP で機能している = CNN 主軸が overkill の可能性
# ユーザー目視で v97m11/v70m2 NG だった反省、 HSV 信用度上げる
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/review_videos/cycle40
mkdir -p logs/board_logs

PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
  --video data/evaluation_videos/v89_match3_95s.mp4 \
  --output data/review_videos/cycle40/cycle40_v89m3.mp4 \
  --cnn-model models/cnn_phase_b_large_v2.pt \
  --hsv-state data/per_video_hsv_ranges/_merged_default.json \
  --dump-board-log logs/board_logs/cycle40_v89m3.jsonl \
  --cnn-override-prob 0.90 \
  > logs/cycle_40_viz_v89m3.log 2>&1

mkdir -p data/verify/cycle40_eval
PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
  --board-log logs/board_logs/cycle40_v89m3.jsonl \
  --report-out data/verify/cycle40_eval/cycle40_v89m3.json \
  > logs/cycle_40_eval_v89m3.log 2>&1

echo "=== cycle 40 DONE @ $(date) ===" | tee logs/cycle_40_done.flag
