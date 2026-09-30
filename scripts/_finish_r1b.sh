#!/usr/bin/env bash
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R1B_PYTHON=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
mkdir -p logs/r1b
nice -n 10 "$R1B_PYTHON" -B -m scripts.run_r1b --step collect > logs/r1b/collect_runner.log 2>&1
nice -n 10 "$R1B_PYTHON" -B -m scripts.run_r1b --step evaluate > logs/r1b/evaluate_runner.log 2>&1
nice -n 10 "$R1B_PYTHON" -B -m scripts.measure_r1b > logs/r1b/measure.log 2>&1
nice -n 10 "$R1B_PYTHON" -B -m scripts.report_r1b > logs/r1b/report.log 2>&1
if "$R1B_PYTHON" -B -c 'import json,sys; sys.exit(not json.load(open("logs/r1b/SUMMARY.json"))["passed"])'; then
    nice -n 10 "$R1B_PYTHON" -B -m scripts.render_r1b_review > logs/r1b/render_runner.log 2>&1
fi
