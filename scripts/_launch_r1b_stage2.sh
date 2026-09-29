#!/usr/bin/env bash
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs/r1b/stage2
R1B_PYTHON=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
nice -n 10 "$R1B_PYTHON" -B -m pytest tests/test_placement_signal_reconcile.py tests/test_placement_signal_runtime.py tests/test_r1_audit.py tests/test_collect_e34c.py tests/test_recognition_pipeline.py tests/test_advantage_overlay_production_recognition_2026-08-13.py -q > logs/r1b/tests.log 2>&1
exec nice -n 10 "$R1B_PYTHON" -B -m scripts.run_r1b_stage2 > logs/r1b/stage2/runner.log 2>&1 < /dev/null
