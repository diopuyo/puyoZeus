#!/bin/bash
# cycle 32d (2026-05-19): cycle 32c の綺麗な seed (= 8 動画 + skip_ojama + review filter)
# で Large CNN を scratch から再学習。 cycle 14 以来の試合外混入バイアスを排除した
# 初の「真に綺麗な seed」 で学習する。
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

VIDEOS="v29m2,v40m7,v51m2,v57m2,v70m2,v89m3,v95m15,v97m11"
OUT="models/cnn_cycle32d.pt"
LOG="logs/cycle_32d_train.log"

echo "=== cycle 32d Large CNN scratch training ==="
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
  --class-balance \
  > "$LOG" 2>&1

echo "=== DONE @ $(date) ===" | tee logs/cycle_32d_done.flag
