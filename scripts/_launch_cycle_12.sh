#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle --cycle 12 --parallel 3 --cnn-override-prob 0.70 --hsv-state data/per_video_hsv_ranges/_merged_default.json > logs/multi_video_cycle_12.log 2>&1
