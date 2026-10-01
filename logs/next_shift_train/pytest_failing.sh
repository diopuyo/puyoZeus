#!/bin/bash
# 使い方: bash pytest_failing.sh <worktree> <出力名>   failing_ids.txt の試験だけ再実行する
cd $1
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
L=/mnt/d/puyo_analyzer/wt_nextfix/logs/next_shift_train
PYTHONPATH=. timeout 3600 /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m pytest -q -p no:cacheprovider -n 6 $(cat $L/failing_ids.txt | tr '\n' ' ') > $L/$2.log 2>&1
echo $? > $L/$2.done
