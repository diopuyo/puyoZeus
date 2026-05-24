#!/bin/bash
# cycle 56_v2 = 真 fine-tune 軽量設定 (= cycle 55 罠回避).
# cycle 55 で「base model 引き継ぎなしの scratch 化」 を実証 → fine-tune バグ修正
# (scripts/phase_i_fine_tune.py の Large 分岐に base load 追加) を活用する真 fine-tune。
#
# 設定:
#   - epochs 2 (= cycle 55 の 12 から大幅削減で過学習回避)
#   - lr 1e-5 (= 1e-4 の 1/10 で軽量、 base 知識保持)
#   - base model: cnn_phase_b_large_v2.pt (= 現 default、 ojama 認識 OK 確認済)
#   - 27 動画 149,523 sample (= cycle 50 final、 100% PURE)
#
# 期待: ojama 認識保持 + 5 色精度小改善
# リスク: cycle 55 同型過学習 (= 軽量設定で回避目標)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p logs

echo "=== cycle 56_v2 training @ $(date) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --store-root data/phase_l/seeds_cycle50_final \
  --all \
  --cell-arch large \
  --cell-base-model models/cnn_phase_b_large_v2.pt \
  --cell-save-to models/cnn_cycle56_v2.pt \
  --epochs 2 \
  --lr 1e-5 \
  --class-balance \
  --augment

echo "=== done @ $(date) ==="
