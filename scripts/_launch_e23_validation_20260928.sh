#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
"$PYTHON" -m pytest tests/test_exchange_event*.py tests/test_e1*.py tests/test_e2*.py -q > logs/e23/tests.log 2>&1
"$PYTHON" -m scripts.run_e23_multilanding_20260928 --source review --off > logs/e23/off.log 2>&1 < /dev/null
"$PYTHON" -m scripts.probe_e23_scene_20260928 > logs/e23/probe.log 2>&1 < /dev/null
