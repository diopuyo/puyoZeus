#!/bin/bash
cd /mnt/d/puyo_analyzer/wt_nextfix
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
PYTHONPATH=. $PY -m scripts.next_shift_m0_cv_20261001 --root logs/next_shift_train/m0_T --check > logs/next_shift_train/m0_check.log 2>&1
echo $? > logs/next_shift_train/m0_check.done
