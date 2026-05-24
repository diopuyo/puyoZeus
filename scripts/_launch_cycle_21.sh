#!/bin/bash
# cycle_21: 物理推論強化 (a+b+d, c は cycle_22 へ切り出し)
#   (a) constraint_valid 永久 False 解消 — 連鎖完了で再有効化
#   (b) 連鎖消費分の tsumo_count 減算 — ChainEvent.before_board から simulate
#   (d) chain_ev 誤発火防止 — OJAMA_FALL / LANDING_GRACE 中の side 別 ban
#
# 2026-05-17 修正: cycle_14 以降の正しい比較には cnn_phase_i_hsv_seed.pt を明示指定が必要。
# multi_video_cycle.py の default は cnn_phase_b_large_v3.pt で別 model のため、
# cycle_19 baseline (mismatch 38) と比較する場合は hsv_seed を明示指定する。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
    --cycle 21 --parallel 3 \
    --cnn-model models/cnn_phase_i_hsv_seed.pt \
    --cnn-override-prob 0.70 \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    > logs/multi_video_cycle_21.log 2>&1
