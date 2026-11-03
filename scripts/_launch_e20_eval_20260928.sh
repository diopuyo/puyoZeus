#!/usr/bin/env bash
set -euo pipefail
cd <home>/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs/e20
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.run_e20_hands_spec_20260928 > logs/e20/eval.log 2>&1 < /dev/null
