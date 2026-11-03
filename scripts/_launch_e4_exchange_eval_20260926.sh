#!/usr/bin/env bash
set -euo pipefail
cd <home>/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. PYTHONUNBUFFERED=1 PYTHONHASHSEED=20260926
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec nice -n 19 <home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.run_e4_exchange_eval_20260926 >> logs/e4/runner.log 2>&1 < /dev/null
