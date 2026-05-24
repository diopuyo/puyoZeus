#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle --cycle 10 --parallel 3 --cnn-override-prob 0.70 > logs/multi_video_cycle_10.log 2>&1
