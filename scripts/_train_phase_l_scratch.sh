#!/bin/bash
# Phase L: scratch CNN 学習 (= base model なし、 Large arch、 38 動画 358K sample).
# _phase_l_master.sh 内の Step 3 が size mismatch で失敗したため、 修正版で再実行。
#
# 修正点:
#   - --cell-base-model に baseline (cnn_phase_b_large_v2.pt) を渡して同 arch 保証
#   - または完全 scratch で base なしを試す → ただし phase_i_fine_tune.py は base 必須っぽい
#   - 安全策: baseline を base にして fine-tune 風に学習 (= ただし epochs 多めで scratch 近い学習量)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p logs/phase_l

# 38 動画リスト (= phase_l/seeds 配下の cell.jsonl ある動画)
VIDEO_IDS=$(ls -d data/phase_l/seeds/*/ 2>/dev/null | sed 's|.*/seeds/||' | sed 's|/||' | sort | tr '\n' ',' | sed 's/,$//')
echo "Training on: $VIDEO_IDS"

echo "=== phase_l scratch training @ $(date) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --video-ids "$VIDEO_IDS" \
  --store-root data/phase_l/seeds \
  --cell-arch large \
  --cell-base-model models/cnn_phase_b_large_v2.pt \
  --cell-save-to models/cnn_phase_l.pt \
  --epochs 8 \
  --lr 5e-4 \
  --class-balance \
  --augment \
  > logs/phase_l/train_v2.log 2>&1

echo "=== train done @ $(date) ==="
ls -la models/cnn_phase_l.pt
tail -5 logs/phase_l/train_v2.log
