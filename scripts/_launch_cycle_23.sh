#!/bin/bash
# cycle_23 multicycle 評価
#   model: cnn_phase_b_crest_v1.pt (CReST oversampling で学習した新 model)
#   cycle_19 baseline (cnn_phase_i_hsv_seed.pt) と比較し、 mismatch/replace 動向を見る
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
    --cycle 23 --parallel 3 \
    --cnn-model models/cnn_phase_b_crest_v1.pt \
    --cnn-override-prob 0.70 \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    > logs/multi_video_cycle_23.log 2>&1
