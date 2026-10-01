#!/bin/bash
cd /mnt/d/puyo_analyzer/wt_nextfix
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=. timeout 7200 /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m pytest -q -p no:cacheprovider -n 6 tests/ > logs/next_shift_train/pytest_all.log 2>&1
echo $? > logs/next_shift_train/pytest_all.done
