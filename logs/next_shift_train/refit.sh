#!/bin/bash
cd /mnt/d/puyo_analyzer/wt_nextfix
export OMP_NUM_THREADS=2
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
L=logs/next_shift_train
PYTHONPATH=. $PY $L/tools/refit_seed.py /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/e15 models/exchange_event_ab_orig_rs1 1 exchange_event_v1 > $L/refit.log 2>&1
PYTHONPATH=. $PY $L/tools/refit_seed.py $L/sprime_T models/exchange_event_ab_T_rs1 1 exchange_event_v5_common >> $L/refit.log 2>&1
echo $? > $L/refit.done
