#!/bin/bash
# cycle 57b (2026-05-23): c56_v3b base + ojama 層凍結 fine-tune.
#
# cycle 57 (baseline base) は ojama 認識消失 (= 4 動画 ratio 0.00%) で失敗。
# 原因: seed に ojama 含まれず conv 中間層が全部書き換わり、 OJAMA row
# 凍結しても 中間特徴量が ojama 用に変わってしまった。
#
# cycle 57b: c56_v3b (= 朝の採用候補、 ojama 102% 維持) を base に再試行。
# 中間層も c56_v3b の「ojama 維持済」 状態から微調整なので退行少ない期待。
#
# 設定: epochs 3、 lr 5e-6 (= さらに軽量、 c56_v3b 状態保持優先)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p logs

echo "=== cycle 57b training @ $(date) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --store-root data/phase_l/seeds_cycle50_final \
  --all \
  --cell-arch large \
  --cell-base-model models/cnn_cycle56_v3b.pt \
  --cell-save-to models/cnn_cycle57b.pt \
  --epochs 3 \
  --lr 5e-6 \
  --class-balance \
  --augment \
  --freeze-ojama-logit \
  > logs/cycle57b_train.log 2>&1

echo "=== train done @ $(date) ==="
ls -la models/cnn_cycle57b.pt
