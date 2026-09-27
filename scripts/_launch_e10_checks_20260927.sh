#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. PYTHONUNBUFFERED=1 PYTHONHASHSEED=20260926
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
"$PYTHON" -m pytest tests/test_exchange_event_*.py tests/test_e3_exchange_metrics.py tests/test_e4_exchange_*.py tests/test_e6_exchange_end.py tests/test_e6b_exchange_score.py tests/test_e8_exchange_finish.py tests/test_e9_exchange_updates.py tests/test_e10_exchange_landing.py tests/test_exchange_virtual_board.py tests/test_scoring.py tests/test_indicators_v2.py -q -p no:xdist > logs/e10/related_tests.log 2>&1
echo 0 > logs/e10/related_tests.exit
"$PYTHON" -c 'from pathlib import Path; from scripts import run_e4_exchange_eval_20260926 as e4; e4.OUT = Path("logs/e10").resolve(); e4.check_off()' > logs/e10/off_check.log 2>&1
echo 0 > logs/e10/off_check.exit
