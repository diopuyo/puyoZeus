#!/bin/bash
# cycle_24 multicycle 評価
#   model: cnn_phase_b_crest_v2.pt (= 19 video CReST 学習 model)
#   cycle_19 baseline (mismatch 38) と cycle_23 (mismatch 48) と比較
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
    --cycle 24 --parallel 3 \
    --cnn-model models/cnn_phase_b_crest_v2.pt \
    --cnn-override-prob 0.70 \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    > logs/multi_video_cycle_24.log 2>&1
