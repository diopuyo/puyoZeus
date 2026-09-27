#!/usr/bin/env bash
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=.:/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=20260926
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
mkdir -p logs/f1b_counter_features
"$PYTHON" -m scripts.train_exchange_event_models_v2_20260927 > logs/f1b_counter_features/runner.log 2>&1 < /dev/null
