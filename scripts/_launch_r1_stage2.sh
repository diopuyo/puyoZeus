#!/usr/bin/env bash
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs/r1/stage2
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
if [ "$#" -gt 0 ]; then
    exec nice -n 10 "$PY" -B -m scripts.run_r1_stage2 --mode "$1" > "logs/r1/stage2/$1.log" 2>&1 < /dev/null
fi
nice -n 10 "$PY" -B -m scripts.run_r1_stage2 --mode off > logs/r1/stage2/off.log 2>&1 < /dev/null &
first=$!
nice -n 10 "$PY" -B -m scripts.run_r1_stage2 --mode on > logs/r1/stage2/on.log 2>&1 < /dev/null &
second=$!
wait "$first"
wait "$second"
