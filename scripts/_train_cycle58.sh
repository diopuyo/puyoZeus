#!/bin/bash
# cycle 58 (= 案 A): 7 クラス seed (= ojama 7064 件含む) で CNN 学習.
# base: baseline (= cnn_phase_b_large_v2.pt)、 軽量 fine-tune。
# 期待: ojama 維持 + 5 色改善 両立 (= cycle 56-57 の罠回避)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p logs

VIDEO_IDS=$(ls -d data/phase_l/seeds_cycle58/*/ 2>/dev/null | sed 's|.*/seeds_cycle58/||' | sed 's|/||' | sort | tr '\n' ',' | sed 's/,$//')
echo "Training on: $VIDEO_IDS"

echo "=== cycle 58 training @ $(date) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --video-ids "$VIDEO_IDS" \
  --store-root data/phase_l/seeds_cycle58 \
  --cell-arch large \
  --cell-base-model models/cnn_phase_b_large_v2.pt \
  --cell-save-to models/cnn_cycle58.pt \
  --epochs 5 \
  --lr 5e-5 \
  --class-balance \
  --augment \
  > logs/cycle58_train.log 2>&1

echo "=== done @ $(date) ==="
tail -5 logs/cycle58_train.log
ls -la models/cnn_cycle58.pt
