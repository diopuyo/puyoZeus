#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
mkdir -p data/indicators_v2/overlay/zap/raw logs/zap_reel
echo "[smoke start] $(date +%s)"
PYTHONPATH=. nice -n 10 ./venv/bin/python -m scripts._zap_1t \
  --video data/frames/video_c5.mp4 \
  --out data/indicators_v2/overlay/zap/raw/_smoke_c5.mp4 \
  --start-sec 1019 --end-sec 1024 --warmup-sec 5 \
  --show-recognition
echo "[smoke end] $(date +%s)"
