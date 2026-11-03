#!/usr/bin/env bash
set -euo pipefail
cd <home>/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs/e24
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.inspect_e24_saved_20260928 > logs/e24/inspect.log 2>&1 < /dev/null
