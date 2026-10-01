#!/bin/bash
# Phase 4 の再生を detach 起動する。使い方: launch_stable.sh <source> <latency> <tag>
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
setsid -f bash -c "$PY -B -m scripts.run_prefire_replay_20260930 --variant stable --source $1 --latency $2 --tag $3 > logs/prefire_prediction/replay/stable$3_$1.log 2>&1 < /dev/null"
