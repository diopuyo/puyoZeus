#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
mkdir -p logs/e15
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -u -m scripts.train_exchange_event_models_v3_20260928 > logs/e15/train.log 2>&1 < /dev/null
