#!/bin/bash
# cycle 44: 試合 11 を 15 秒前から切り出し (= 862-966 秒、 104 秒)
# 認識 pipeline ウォームアップ用バッファ確保、 ユーザー指摘「14 秒で品質 OK」 検証
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/holdout_videos
mkdir -p data/review_videos/cycle44
mkdir -p logs/board_logs
mkdir -p data/verify/cycle44_eval

# Step 1: ffmpeg で試合 11 を 862 秒から 104 秒切り出し
FFMPEG=$(./venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
echo "[step1] 試合 11 + 15 秒バッファ 切り出し開始 @ $(date)"
"$FFMPEG" -y -ss 862 -i data/frames/video_30.mp4 -t 104 -c copy \
  data/holdout_videos/v30_match11_buf15s.mp4 > logs/cycle_44_cut.log 2>&1
echo "[step1] 切り出し完了 @ $(date)"
ls -la data/holdout_videos/v30_match11_buf15s.mp4

# Step 2: viz
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
  --video data/holdout_videos/v30_match11_buf15s.mp4 \
  --output data/review_videos/cycle44/cycle44_v30_match11_buf15.mp4 \
  --cnn-model models/cnn_phase_b_large_v2.pt \
  --hsv-state data/per_video_hsv_ranges/_merged_default.json \
  --dump-board-log logs/board_logs/cycle44_v30_match11_buf15.jsonl \
  > logs/cycle_44_viz.log 2>&1
echo "[step2] viz 完了 @ $(date)"

# Step 3: 評価
PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
  --board-log logs/board_logs/cycle44_v30_match11_buf15.jsonl \
  --report-out data/verify/cycle44_eval/cycle44_v30_match11_buf15.json \
  > logs/cycle_44_eval.log 2>&1

echo "=== cycle 44 DONE @ $(date) ===" | tee logs/cycle_44_done.flag
