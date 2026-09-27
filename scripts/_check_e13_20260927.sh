#!/usr/bin/env bash
set -u
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. PYTHONUNBUFFERED=1 PYTHONHASHSEED=20260926
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
"$PYTHON" -m pytest tests/test_exchange_event_*.py tests/test_e3_exchange_metrics.py tests/test_e4_exchange_*.py tests/test_e6_exchange_end.py tests/test_e6b_exchange_score.py tests/test_e8_exchange_finish.py tests/test_e9_exchange_updates.py tests/test_e10_exchange_landing.py tests/test_e10b_exchange_landing.py tests/test_e10c_exchange_landing.py tests/test_e11_review_data_panel.py tests/test_e12_exchange_evaluation.py tests/test_e12b_exchange_completion.py tests/test_e13_frame_evaluation.py tests/test_exchange_virtual_board.py tests/test_scoring.py tests/test_indicators_v2.py -q -p no:xdist > logs/e13/related_tests.log 2>&1
echo $? > logs/e13/related_tests.exit
"$PYTHON" logs/e13/check_sha.py > logs/e13/sha.log 2>&1
echo $? > logs/e13/sha.exit
