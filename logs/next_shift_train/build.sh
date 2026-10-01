#!/bin/bash
cd /mnt/d/puyo_analyzer/wt_nextfix
PYTHONPATH=. /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.next_shift_build_queues_20261001 > logs/next_shift_train/build.log 2>&1
echo $? > logs/next_shift_train/build.done
