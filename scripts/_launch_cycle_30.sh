#!/bin/bash
# cycle_30 multicycle 評価 (bg_fp の HybridClassifier 伝達)
#   背景指紋 (bg_fp) を HybridClassifier に渡し、 各 cell の HSV と背景指紋の
#   距離 < SOFT_THRESHOLD (= 70) なら、 CNN が puyo を提案しても empty 強制。
#   v97 / v50 で大量発生中の青背景バイアス (= CNN が青背景を「青 puyo」 と誤分類)
#   の根本対策。
#   model: cnn_phase_i_hsv_seed.pt (= cycle_19 baseline と同条件)
#   cycle 29 と比較して背景誤認削減を確認
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
    --cycle 30 --parallel 3 \
    --cnn-model models/cnn_phase_i_hsv_seed.pt \
    --cnn-override-prob 0.70 \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    > logs/multi_video_cycle_30.log 2>&1
