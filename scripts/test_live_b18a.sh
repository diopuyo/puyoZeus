#!/usr/bin/env bash
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/rt/puyo_analyzer
mkdir -p logs/live_b18a
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
exec nice -n 10 /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m pytest \
  tests/phase_j tests/test_live*.py tests/test_exchange_event*.py tests/test_e[0-9]*.py \
  tests/test_production_config.py tests/test_measure_realtime_breakdown_20260928.py \
  tests/test_run_live_pipeline_20260928.py \
  -q --junitxml=logs/live_b18a/tests.xml > logs/live_b18a/tests.log 2>&1 < /dev/null
