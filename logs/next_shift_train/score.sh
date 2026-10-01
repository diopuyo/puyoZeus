#!/bin/bash
cd /mnt/d/puyo_analyzer/nextfix_run
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=/mnt/d/puyo_analyzer/wt_nextfix /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python /mnt/d/puyo_analyzer/wt_nextfix/logs/next_shift_train/tools/score.py $1 > /mnt/d/puyo_analyzer/wt_nextfix/logs/next_shift_train/score_$1.txt 2>&1
echo $? >> /mnt/d/puyo_analyzer/wt_nextfix/logs/next_shift_train/score_$1.txt
