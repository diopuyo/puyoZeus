#!/bin/bash
# cycle 32e (2026-05-19): EMPTY 追加 seed で Large CNN scratch 学習。
# cycle 32d との差分:
#  - EMPTY を 7 クラス目に追加 (= 0 ラベル sample 3,480 件)
#  - class_balance OFF (= cycle 15 の empty dominant 副作用回避)
#  - ojama は依然 train data ゼロ (= skip)、 推論時に logit mask で対処
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

VIDEOS="v29m2,v40m7,v51m2,v57m2,v70m2,v89m3,v95m15,v97m11"
OUT="models/cnn_cycle32e.pt"
LOG="logs/cycle_32e_train.log"

echo "=== cycle 32e Large CNN scratch training (EMPTY追加) ==="
echo "videos: $VIDEOS"
echo "output: $OUT"
echo "log:    $LOG"
echo "started: $(date)"

PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --video-ids "$VIDEOS" \
  --store-root data/pseudo_labels_hsv_seed \
  --apply-review-filter \
  --cell-arch large \
  --epochs 5 \
  --lr 1e-3 \
  --cell-save-to "$OUT" \
  --augment \
  > "$LOG" 2>&1

echo "=== DONE @ $(date) ===" | tee logs/cycle_32e_train_done.flag
