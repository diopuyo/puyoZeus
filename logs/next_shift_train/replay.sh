#!/bin/bash
# 使い方: bash replay.sh <variant> <source>   (cwd は exev の logs を読める run ディレクトリ)
cd /mnt/d/puyo_analyzer/nextfix_run
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
L=/mnt/d/puyo_analyzer/wt_nextfix/logs/next_shift_train/replay_logs
mkdir -p $L
PYTHONPATH=/mnt/d/puyo_analyzer/wt_nextfix $PY -B -m scripts.next_shift_replay_20261001 --variant $1 --source $2 > $L/$1_$2.log 2>&1
echo $? > $L/$1_$2.done
