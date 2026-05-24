#!/bin/bash
# cycle_20: bg_fp 構造改革
#   - NextDetector トリガーで採取タイミングを CNN 非依存化
#   - cell 画像 patch を pattern として保存 (HSV mean → 画像差分)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
    --cycle 20 --parallel 3 \
    --cnn-override-prob 0.70 \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    > logs/multi_video_cycle_20.log 2>&1
