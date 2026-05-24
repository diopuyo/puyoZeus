#!/bin/bash
# parallel_bulk.py 完了後に統合+CNN学習
set -e
LOG=/tmp/wait_parallel_train.log
echo "[$(date)] 待機開始" > $LOG
while pgrep -f "scripts/parallel_bulk.py" > /dev/null 2>&1; do
  sleep 180
done
echo "[$(date)] parallel_bulk 終了、merge+train 実行" >> $LOG
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
source venv/bin/activate
PYTHONPATH=. python scripts/merge_parallel_and_train.py >> $LOG 2>&1
echo "[$(date)] 全完了" >> $LOG
