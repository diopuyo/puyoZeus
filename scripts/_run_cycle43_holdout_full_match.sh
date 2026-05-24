#!/bin/bash
# cycle 43: video_30 試合 11 (= 89 秒) 完全切り出し holdout 評価
# 試合開始から終了まで完全に含む = ユーザー指示「試合切り抜き必要」 対応
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/holdout_videos
mkdir -p data/review_videos/cycle43
mkdir -p logs/board_logs
mkdir -p data/verify/cycle43_eval

# Step 1: ffmpeg で試合 11 切り出し (= 877-966 秒)
FFMPEG=$(./venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
echo "[step1] ffmpeg 試合 11 切り出し開始 @ $(date)"
"$FFMPEG" -y -ss 877 -i data/frames/video_30.mp4 -t 89 -c copy \
  data/holdout_videos/v30_match11_89s.mp4 > logs/cycle_43_cut.log 2>&1
echo "[step1] 切り出し完了 @ $(date)"
ls -la data/holdout_videos/v30_match11_89s.mp4

# Step 2: viz (= 既 default model + 現状設定)
echo "[step2] viz 開始 @ $(date)"
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
  --video data/holdout_videos/v30_match11_89s.mp4 \
  --output data/review_videos/cycle43/cycle43_v30_match11.mp4 \
  --cnn-model models/cnn_phase_b_large_v2.pt \
  --hsv-state data/per_video_hsv_ranges/_merged_default.json \
  --dump-board-log logs/board_logs/cycle43_v30_match11.jsonl \
  > logs/cycle_43_viz.log 2>&1
echo "[step2] viz 完了 @ $(date)"

# Step 3: 評価
PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
  --board-log logs/board_logs/cycle43_v30_match11.jsonl \
  --report-out data/verify/cycle43_eval/cycle43_v30_match11.json \
  > logs/cycle_43_eval.log 2>&1

echo "=== cycle 43 DONE @ $(date) ===" | tee logs/cycle_43_done.flag
