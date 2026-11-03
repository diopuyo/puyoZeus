#!/usr/bin/env bash
set -euo pipefail
cd <home>/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m pytest tests/test_exchange_event*.py tests/test_e1*.py tests/test_e2*.py -q > logs/e22/tests.log 2>&1
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.run_e22_death_formula_20260928 --source review --off > logs/e22/off.log 2>&1 < /dev/null
