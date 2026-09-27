#!/usr/bin/env bash
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=.:/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=20260926
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
mkdir -p logs/e14
"$PYTHON" -m pytest tests/test_exchange_event_evaluator.py tests/test_exchange_event_overlay.py tests/test_exchange_event_tracker.py tests/test_exchange_event_m0.py tests/test_f1_counter_features.py tests/test_f1b_temporal_features.py tests/test_e14_count_features.py -q > logs/e14/initial_tests.log 2>&1
"$PYTHON" -m scripts.run_e14_exchange_replay_20260927 > logs/e14/runner.log 2>&1
