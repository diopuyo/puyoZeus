#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs/e20
/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.render_e20_review_20260928 --variant "${1:-e19}" > logs/e20/review_${1:-e19}.log 2>&1 < /dev/null
