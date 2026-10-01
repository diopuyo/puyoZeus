#!/usr/bin/env bash
# 使い方: _run_switch_report.sh <variant> [baseline]
set -eu
cd /mnt/d/puyo_analyzer/wt_switch
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
nice -n 10 $PY -B -m scripts.report_switch_smoothing --variant $1 --baseline ${2:-off}
