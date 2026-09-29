#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
mkdir -p logs/e16/train
export PYTHONPATH=.
/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -u -m scripts.train_exchange_event_models_v4_20260928 > logs/e16/train/train.log 2>&1 < /dev/null
