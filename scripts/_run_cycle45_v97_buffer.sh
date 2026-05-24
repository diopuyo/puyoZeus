#!/bin/bash
# cycle 45: video_97 試合 11 を 15 秒バッファ付き再切り出し
# 既存 v97_match11_96s.mp4 はバッファなし = ユーザー指摘「ぷよ→empty 誤認多数」 の原因疑い
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/holdout_videos
mkdir -p data/review_videos/cycle45
mkdir -p logs/board_logs
mkdir -p data/verify/cycle45_eval

# Step 1: ffmpeg で切り出し (= 1898-2009 秒、 111 秒)
FFMPEG=$(./venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
echo "[step1] v97 試合 11 + 15 秒バッファ 切り出し開始 @ $(date)"
"$FFMPEG" -y -ss 1898 -i data/frames/video_97.webm -t 111 \
  -c:v libx264 -preset ultrafast -an \
  data/holdout_videos/v97_match11_buf15s.mp4 > logs/cycle_45_cut.log 2>&1
echo "[step1] 切り出し完了 @ $(date)"
ls -la data/holdout_videos/v97_match11_buf15s.mp4

# Step 2: viz
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
  --video data/holdout_videos/v97_match11_buf15s.mp4 \
  --output data/review_videos/cycle45/cycle45_v97_match11_buf15.mp4 \
  --cnn-model models/cnn_phase_b_large_v2.pt \
  --hsv-state data/per_video_hsv_ranges/_merged_default.json \
  --dump-board-log logs/board_logs/cycle45_v97_match11_buf15.jsonl \
  > logs/cycle_45_viz.log 2>&1
echo "[step2] viz 完了 @ $(date)"

# Step 3: 評価
PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
  --board-log logs/board_logs/cycle45_v97_match11_buf15.jsonl \
  --report-out data/verify/cycle45_eval/cycle45_v97_match11_buf15.json \
  > logs/cycle_45_eval.log 2>&1

echo "=== cycle 45 DONE @ $(date) ===" | tee logs/cycle_45_done.flag
