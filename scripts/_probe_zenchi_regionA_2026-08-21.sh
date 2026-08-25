#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_advantage_overlay \
  --video data/frames/video_zenchi_c0BQoMJwwQU.mp4 \
  --start-sec 0 --end-sec 300 \
  --layout panel --show-recognition --no-force-in-match \
  --model-dir data/verify/retrain_model62_2026-08-21 \
  --out data/verify/zenchi_probe_2026-08-21/regionA_0_300.mp4
