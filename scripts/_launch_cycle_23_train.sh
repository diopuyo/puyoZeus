#!/bin/bash
# cycle_23 学習: CReST oversampling で「cycle_14 puyo best + empty 学習」 を両立
#   - base: cnn_phase_i_hsv_seed.pt (cycle_14 神 model)
#   - seed: pseudo_labels_hsv_seed_with_empty (14,000 件 = 5 色 11,500 + empty 2,500)
#   - class_balance OFF: cycle_15 で empty dominant 化した失敗を回避
#   - oversample-alpha 0.5: minority (puyo 各色) を頻繁採択 → puyo 認識力強化
#   - focal+logit: noise + imbalance 両対処
#   - augment: 4 色 permutation
#   - epochs 10: CPU/GPU リソース活用方針 (default 5 から増加)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
    --component cell_color --all \
    --store-root data/pseudo_labels_hsv_seed_with_empty \
    --cell-base-model models/cnn_phase_i_hsv_seed.pt \
    --cell-save-to models/cnn_phase_b_crest_v1.pt \
    --cell-arch large \
    --epochs 10 --augment \
    --oversample-alpha 0.5 \
    --focal-gamma 2.0 \
    --logit-adjust-tau 1.0 \
    > logs/cycle_23_train.log 2>&1
