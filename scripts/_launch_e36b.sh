#!/usr/bin/env bash
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
# (a) E36b を先に、続けて (b) D5b を1本ずつ順に走らせる。
nice -n 10 "$PY" -B -m scripts.run_e36b > logs/e36b/runner.log 2>&1 < /dev/null
nice -n 10 "$PY" -B -m scripts.run_d5b > logs/d5b/runner.log 2>&1 < /dev/null
