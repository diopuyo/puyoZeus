#!/bin/bash
# cycle 57 (2026-05-23): 5 色 hard sample fine-tune + ojama 層凍結.
#
# 設計:
#   - base: cnn_phase_b_large_v2.pt (= baseline)
#   - seed: cycle 50 final (= 27 動画 149,523 sample、 100% PURE)
#   - --freeze-ojama-logit: 最終層 OJAMA row 凍結 (= 朝の c56_v2 退行回避)
#   - epochs 5、 lr 1e-5、 class_balance、 augment
#
# 期待: 5 色精度改善 + ojama 認識完全維持
# リスク: cycle 56 系と類似の「ぎりぎり改善 or 逆効果」 (= ただし新軸 = 凍結)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p logs

echo "=== cycle 57 training @ $(date) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --store-root data/phase_l/seeds_cycle50_final \
  --all \
  --cell-arch large \
  --cell-base-model models/cnn_phase_b_large_v2.pt \
  --cell-save-to models/cnn_cycle57.pt \
  --epochs 5 \
  --lr 1e-5 \
  --class-balance \
  --augment \
  --freeze-ojama-logit \
  > logs/cycle57_train.log 2>&1

echo "=== train done @ $(date) ==="
ls -la models/cnn_cycle57.pt
