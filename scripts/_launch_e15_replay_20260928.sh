#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -u -m scripts.run_e15_exchange_replay_20260928 --all > logs/e15/replay.log 2>&1 < /dev/null
