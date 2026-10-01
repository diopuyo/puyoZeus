#!/bin/bash
# Phase 3 最善手の再生を detach 起動する。使い方: launch_bestplay.sh <source> <latency> <tag> [evaluator=full]
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
EVAL=${4:-full}
setsid -f bash -c "$PY -B -m scripts.run_prefire_replay_20260930 --variant bestplay --source $1 --latency $2 --tag $3 --evaluator $EVAL > logs/prefire_prediction/replay/bestplay$3_$1.log 2>&1 < /dev/null"
