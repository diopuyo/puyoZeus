#!/bin/bash
cd /mnt/d/puyo_analyzer/wt_nextfix
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
PYTHONPATH=. $PY -m scripts.next_shift_retrain_sprime_20261001 --variant T --output logs/next_shift_train/sprime_T --models models/exchange_event_v5 --workers 6 > logs/next_shift_train/sprime_T.log 2>&1
echo $? > logs/next_shift_train/sprime_T.done
