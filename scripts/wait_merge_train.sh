#!/bin/bash
# PL1 + PL2 両方の bulk_train が終わるまで待って統合CNN学習
set -e

LOG=/tmp/wait_merge_train.log
echo "[$(date)] Waiting for both bulk_train processes" > $LOG

# どちらも終わるまで待つ
while pgrep -f "scripts/bulk_train" > /dev/null 2>&1; do
  sleep 180
done
echo "[$(date)] Both finished" >> $LOG

cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
source venv/bin/activate

# 最新のPL1, PL2 データセットを merge
PYTHONPATH=. python scripts/merge_and_train.py >> $LOG 2>&1
echo "[$(date)] Merge+Train done" >> $LOG
