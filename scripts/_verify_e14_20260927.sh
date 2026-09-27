#!/usr/bin/env bash
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=.:/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=20260926
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
while [ ! -f logs/e14/report.json ]; do sleep 10; done
set +e
"$PYTHON" -m pytest tests/test_exchange_event*.py tests/test_e[0-9]*.py tests/test_f1*.py tests/test_indicators_v2.py tests/test_advantage_overlay*.py -q > logs/e14/related_tests.log 2>&1
code=$?
printf '%s\n' "$code" > logs/e14/related_tests.exit
exit "$code"
