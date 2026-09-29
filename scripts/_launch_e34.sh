#!/usr/bin/env bash
set -eu
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
$PY -m scripts.run_e34 --preflight > logs/e34/preflight.log 2>&1
$PY -m scripts.run_e34 > logs/e34/runner.log 2>&1 < /dev/null
