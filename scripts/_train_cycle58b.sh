#!/bin/bash
# cycle 58b (= 真因対策): --class-balance を外して再学習.
#
# cycle 58 失敗原因: class_balance で OJAMA 4.64 倍 + EMPTY 3.82 倍 の過剰重みで
# 5 色精度が崩壊 (= +62% 悪化)。 重みを全 1 倍に戻して 5 色精度回復期待。
# trade-off: ojama 学習弱まる (= 7K/174K = 4% の比率で学習) ため、 認識率の
# 自然な低下は予想されるが、 cycle 58 のような「5 色 → OJAMA flip 多発」 は防げる。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p logs

VIDEO_IDS=$(ls -d data/phase_l/seeds_cycle58/*/ 2>/dev/null | sed 's|.*/seeds_cycle58/||' | sed 's|/||' | sort | tr '\n' ',' | sed 's/,$//')

echo "=== cycle 58b training @ $(date) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --video-ids "$VIDEO_IDS" \
  --store-root data/phase_l/seeds_cycle58 \
  --cell-arch large \
  --cell-base-model models/cnn_phase_b_large_v2.pt \
  --cell-save-to models/cnn_cycle58b.pt \
  --epochs 5 \
  --lr 5e-5 \
  --augment \
  > logs/cycle58b_train.log 2>&1

echo "=== done @ $(date) ==="
tail -5 logs/cycle58b_train.log
ls -la models/cnn_cycle58b.pt
