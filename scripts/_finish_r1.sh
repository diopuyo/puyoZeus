#!/usr/bin/env bash
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R1_PYTHON=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
mkdir -p logs/r1
nice -n 10 "$R1_PYTHON" -B -m pytest tests/test_placement_signal_reconcile.py tests/test_placement_signal_runtime.py tests/test_r1_audit.py tests/test_collect_e34c.py -q > logs/r1/tests.log 2>&1
# 先行収集が完全に終わってから、未完了分のみを再開する。
if [ "$#" -gt 0 ]; then
    while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi
nice -n 10 "$R1_PYTHON" -B -m scripts.run_r1_collect >> logs/r1/collect_runner.log 2>&1
nice -n 10 "$R1_PYTHON" -B -m scripts.run_r1_evaluate > logs/r1/evaluate_runner.log 2>&1
nice -n 10 "$R1_PYTHON" -B -m scripts.measure_r1_recognition > logs/r1/measure.log 2>&1
nice -n 10 "$R1_PYTHON" -B -m scripts.report_r1 > logs/r1/report.log 2>&1
if "$R1_PYTHON" -B -c 'import json,sys; sys.exit(not json.load(open("logs/r1/SUMMARY.json"))["passed"])'; then
    nice -n 10 "$R1_PYTHON" -B -m scripts.render_r1_review > logs/r1/render_runner.log 2>&1
fi
