#!/bin/bash
cd /mnt/d/puyo_analyzer/wt_nextfix
PYTHONPATH=. /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.next_shift_measure_20261001 > logs/next_shift_train/measure.json 2> logs/next_shift_train/measure.err
echo done > logs/next_shift_train/measure.done
