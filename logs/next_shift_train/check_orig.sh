#!/bin/bash
cd /mnt/d/puyo_analyzer/wt_nextfix
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
PYTHONPATH=. /usr/bin/time -v $PY -m scripts.next_shift_retrain_sprime_20261001 --variant orig --output logs/next_shift_train/sprime_orig_check --videos video_29 video_c1 video_c50 --workers 3 > logs/next_shift_train/check_orig.log 2>&1
echo $? > logs/next_shift_train/check_orig.done
