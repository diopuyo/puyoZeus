#!/usr/bin/env bash
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec nice -n 10 /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -B -m scripts.run_r1_collect >> logs/r1/collect_runner.log 2>&1 < /dev/null
