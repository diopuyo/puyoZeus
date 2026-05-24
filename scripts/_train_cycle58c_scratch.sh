#!/bin/bash
# cycle 58c (= 2026-05-23 23:00 着手): scratch 学習で baseline 越え狙い.
#
# 今夜 fine-tune 5 連敗確認 (= cycle 57/57b/Phase L/58/58b)。
# baseline (= cnn_phase_b_large_v2.pt = cycle 71v) は scratch 学習で
# ojama 完璧 + 5 色「ちょっと残る」 達成済。
# 同じ作り方を cycle 58 seed (= 30 動画 195K cells、 ojama 7K) で再現。
#
# 設定 (= baseline と同じ):
#   - base: なし (= --cell-base-model に存在しないダミーパス渡し scratch 動作)
#   - epoch 15、 lr 1e-3、 class_balance、 augment、 arch large
#
# 期待: ojama 完璧維持 + 5 色精度 改善 (= データ 6 倍効果)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p logs

VIDEO_IDS=$(ls -d data/phase_l/seeds_cycle58/*/ 2>/dev/null | sed 's|.*/seeds_cycle58/||' | sed 's|/||' | sort | tr '\n' ',' | sed 's/,$//')
echo "Training on: $VIDEO_IDS"

echo "=== cycle 58c scratch training @ $(date) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --video-ids "$VIDEO_IDS" \
  --store-root data/phase_l/seeds_cycle58 \
  --cell-arch large \
  --cell-base-model models/__scratch_sentinel_does_not_exist__.pt \
  --cell-save-to models/cnn_cycle58c.pt \
  --epochs 15 \
  --lr 1e-3 \
  --class-balance \
  --augment \
  > logs/cycle58c_train.log 2>&1

echo "=== done @ $(date) ==="
tail -5 logs/cycle58c_train.log
ls -la models/cnn_cycle58c.pt
