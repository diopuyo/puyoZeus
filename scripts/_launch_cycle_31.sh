#!/bin/bash
# cycle_31 multicycle 評価 (B 軸: state machine 堅牢化 = baseline 自己修復)
#   STABLE 中に baseline と CNN puyo 数 diff が連続異常 (= |diff|>8 を 60 frame
#   = 1 秒連続) なら baseline 壊れ判定 → state reset (= MENU 戻し + bg_fp 再採取)
#   v97 53 秒 TSUMO_FALL 詰まり問題への救済策。
#   cycle 30 (bg_fp HybridClassifier 連携) は撤去済 → cycle 29 状態 + B 軸
#   model: cnn_phase_i_hsv_seed.pt (= cycle_19 baseline と同条件)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
    --cycle 31 --parallel 3 \
    --cnn-model models/cnn_phase_i_hsv_seed.pt \
    --cnn-override-prob 0.70 \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    > logs/multi_video_cycle_31.log 2>&1
