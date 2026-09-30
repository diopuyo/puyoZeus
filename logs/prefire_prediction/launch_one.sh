#!/bin/bash
# 1記録の再生を nice 0 で detach 起動する。使い方: launch_one.sh <variant> <source> [--compare]
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
setsid -f bash -c "nice -n 0 $PY -B -m scripts.run_prefire_replay_20260930 --variant $1 --source $2 $3 > logs/prefire_prediction/replay/$1_$2.log 2>&1 < /dev/null"
