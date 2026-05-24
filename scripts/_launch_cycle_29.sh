#!/bin/bash
# cycle_29 multicycle 評価 (着地起点を NEXT 移動に変更)
#   1. grace 期間 12→5 frame 短縮
#   2. grace + landing_vote 起動を「TSUMO_FALL→STABLE」 から「NEXT 移動検知」 に変更
#      → state machine 詰まり (v97 53 秒問題) を救済
#   3. 案 X / 案 X 改 削除 (= 副作用大)
#   4. H2+H3 (連鎖後 constraint 復活 + tsumo 減算) 維持
#   model: cnn_phase_i_hsv_seed.pt (= cycle_19 baseline と同条件)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
    --cycle 29 --parallel 3 \
    --cnn-model models/cnn_phase_i_hsv_seed.pt \
    --cnn-override-prob 0.70 \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    > logs/multi_video_cycle_29.log 2>&1
