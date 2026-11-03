#!/usr/bin/env bash
set -eu
cd <home>/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=.:<home>/.gemini/antigravity/scratch/puyo_analyzer
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=20260926
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
mkdir -p logs/f1b_counter_features
"$PYTHON" -m scripts.audit_retrain_f1b_20260927 > logs/f1b_counter_features/runner.log 2>&1 < /dev/null
