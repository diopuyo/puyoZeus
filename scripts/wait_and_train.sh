#!/bin/bash
# bulk_train.py 完了を待って、自動的に CNN 学習+可視化を実行する
set -e

BULK_PID=34562
LOG=/tmp/wait_and_train.log
echo "[$(date)] Waiting for bulk_train (PID $BULK_PID)" > $LOG

# bulk_train プロセスが消えるまで待つ
while kill -0 $BULK_PID 2>/dev/null; do
  sleep 120
done
echo "[$(date)] bulk_train finished" >> $LOG

cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
source venv/bin/activate

# 最新の bulk_patches_balanced_through_vNN.npz を特定
LATEST=$(ls -t data/training/bulk_patches_balanced_through_v*.npz 2>/dev/null | head -1)
if [[ -z "$LATEST" ]]; then
  echo "[$(date)] ERROR: no bulk patches found" >> $LOG
  exit 1
fi
echo "[$(date)] Training on: $LATEST" >> $LOG

# CNN 学習+可視化
PYTHONPATH=. python scripts/train_final_cnn.py "$LATEST" >> $LOG 2>&1
echo "[$(date)] Training done" >> $LOG
