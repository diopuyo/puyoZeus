#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
mkdir -p logs/e19/train
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.train_e19_counter_probability_20260928 > logs/e19/train/train.log 2>&1 < /dev/null
