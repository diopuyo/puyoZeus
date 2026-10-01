#!/bin/bash
cd /mnt/d/puyo_analyzer/wt_nextfix
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
L=logs/next_shift_train
PYTHONPATH=. nice -n 10 $PY $L/tools/s1_cv.py /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/e15 $L/s1cv_orig > $L/s1cv_orig.log 2>&1
PYTHONPATH=. nice -n 10 $PY $L/tools/s1_cv.py $L/sprime_T $L/s1cv_T > $L/s1cv_T.log 2>&1
PYTHONPATH=. nice -n 10 $PY $L/tools/paired_cv.py $L/s1cv_orig $L/s1cv_T 'seed_*/fold_*/predictions.csv' S1_prime S1_prime_light > $L/sprime_T/PAIRED_CV_S1.json 2>> $L/s1cv_T.log
echo $? > $L/s1cv.done
