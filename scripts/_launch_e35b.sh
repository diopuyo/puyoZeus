#!/usr/bin/env bash
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
E35B_PYTHON=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
mkdir -p logs/e35b
for source in review q_7gc4TgFig fcXG83vInDY; do
    nice -n 10 "$E35B_PYTHON" -B -m scripts.measure_e35b --source "$source" > "logs/e35b/$source.log" 2>&1 &
done
wait
for source in mia8KCjr52g zenchi; do
    nice -n 10 "$E35B_PYTHON" -B -m scripts.measure_e35b --source "$source" > "logs/e35b/$source.log" 2>&1 &
done
wait
