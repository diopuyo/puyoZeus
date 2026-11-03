#!/usr/bin/env bash
set -eu
cd <home>/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. PYTHONUNBUFFERED=1 PYTHONHASHSEED=20260926
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export E12_OUT=logs/e13
PYTHON=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
"$PYTHON" logs/e12/run.py > logs/e13/runner.log 2>&1
