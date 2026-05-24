#!/bin/bash
# cycle 42: 真 holdout 動画切り出し + 評価
# video_30 (= 学習未使用、 light vs あん マスター級) の 5-6.5 分目を 90 秒切り出し
# 既 default model + cycle 33 設定 (= tier1<20、 boost OFF) で viz 評価
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/holdout_videos
mkdir -p data/review_videos/cycle42
mkdir -p logs/board_logs
mkdir -p data/verify/cycle42_eval

# Step 1: ffmpeg で切り出し (5 分目から 90 秒)
FFMPEG=$(./venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
echo "[step1] ffmpeg holdout 切り出し開始 @ $(date)"
"$FFMPEG" -y -ss 300 -i data/frames/video_30.mp4 -t 90 -c copy \
  data/holdout_videos/v30_5min_90s.mp4 > logs/cycle_42_cut.log 2>&1
echo "[step1] holdout 切り出し完了 @ $(date)"
ls -la data/holdout_videos/v30_5min_90s.mp4

# Step 2: viz (= 既 default model + 現状の image_reader/hybrid_classifier 設定)
echo "[step2] viz 開始 @ $(date)"
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
  --video data/holdout_videos/v30_5min_90s.mp4 \
  --output data/review_videos/cycle42/cycle42_v30_holdout.mp4 \
  --cnn-model models/cnn_phase_b_large_v2.pt \
  --hsv-state data/per_video_hsv_ranges/_merged_default.json \
  --dump-board-log logs/board_logs/cycle42_v30_holdout.jsonl \
  > logs/cycle_42_viz.log 2>&1
echo "[step2] viz 完了 @ $(date)"

# Step 3: 評価
PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
  --board-log logs/board_logs/cycle42_v30_holdout.jsonl \
  --report-out data/verify/cycle42_eval/cycle42_v30_holdout.json \
  > logs/cycle_42_eval.log 2>&1

echo "=== cycle 42 DONE @ $(date) ===" | tee logs/cycle_42_done.flag
