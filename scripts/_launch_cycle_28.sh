#!/bin/bash
# cycle_28 multicycle 評価 (= cycle_27 + H2 + H3 統合)
#   H2: 連鎖完了で constraint_valid 再有効化
#   H3: ChainSimulator chain_result から消去 puyo 色を集計 → tsumo_count 減算
#   = cycle_21 の (a+b) を viz 評価込みで再検証
#   model: cnn_phase_i_hsv_seed.pt (= cycle_19 baseline と同条件)
#   cycle_27 と比較して連鎖後の背景誤認 / 認識補正効果を確認
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
    --cycle 28 --parallel 3 \
    --cnn-model models/cnn_phase_i_hsv_seed.pt \
    --cnn-override-prob 0.70 \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    > logs/multi_video_cycle_28.log 2>&1
