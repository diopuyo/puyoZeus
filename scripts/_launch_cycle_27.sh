#!/bin/bash
# cycle_27 multicycle 評価 (= cycle_26 (A1+A2+A4) + 案 X 統合)
#   案 X: tsumo_total == 0 の STABLE で field 強制 empty
#         → 試合開始時 / 試合切替直後の背景誤認を物理推論で排除
#   model: cnn_phase_i_hsv_seed.pt (= cycle_19 baseline と同条件)
#   cycle_19 / cycle_26 と比較して、 試合開始時の背景誤認削減を確認
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
    --cycle 27 --parallel 3 \
    --cnn-model models/cnn_phase_i_hsv_seed.pt \
    --cnn-override-prob 0.70 \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    > logs/multi_video_cycle_27.log 2>&1
