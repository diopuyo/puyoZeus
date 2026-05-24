#!/bin/bash
# cycle_26 multicycle 評価 (= A1+A2+A4 統合実装、 着地直後の誤認削減)
#   A1: grace 中 confirmed_board 完全凍結 (constraint/vote/long-term override skip)
#   A2: LANDING_VOTE 初期 5 frame 除外 + NEXT 色不一致時 ratio 0.5
#   A4: NEXT 色 prior 強化 (ratio>=0.7) + 早期確定 (len>=5, ratio>=0.8)
#   model: cnn_phase_i_hsv_seed.pt (= cycle_19 baseline と同条件)
#   cycle_19 (mismatch 38) と比較して、 viz 上の「置いた直後の誤認」 削減を検証
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
    --cycle 26 --parallel 3 \
    --cnn-model models/cnn_phase_i_hsv_seed.pt \
    --cnn-override-prob 0.70 \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    > logs/multi_video_cycle_26.log 2>&1
