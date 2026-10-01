#!/bin/bash
cd /mnt/d/puyo_analyzer/wt_nextfix
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
PYTHONPATH=. /usr/bin/time -v $PY -m scripts.next_shift_m0_inputs_20261001 --variant T --output logs/next_shift_train/m0_T > logs/next_shift_train/m0_inputs.log 2>&1
echo $? > logs/next_shift_train/m0_inputs.done
