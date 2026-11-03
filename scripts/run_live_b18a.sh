#!/usr/bin/env bash
set -eu
cd <home>/.codex/worktrees/rt/puyo_analyzer
mkdir -p logs/live_b18a
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
exec nice -n 10 <home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -u -m scripts.verify_live_b18a "$@" > logs/live_b18a/runner.log 2>&1 < /dev/null
