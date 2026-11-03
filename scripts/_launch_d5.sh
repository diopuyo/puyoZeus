#!/usr/bin/env bash
set -eu
cd <home>/.codex/worktrees/exev/puyo_analyzer
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D5_PYTHON=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
exec nice -n 10 "$D5_PYTHON" -B -m scripts.run_d5 > logs/d5/runner.log 2>&1 < /dev/null
