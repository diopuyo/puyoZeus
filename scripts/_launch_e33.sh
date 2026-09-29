#!/bin/bash
set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
$PY -m pytest tests/test_e33_stage_timeout.py tests/test_e31_prefire_snapshot.py tests/test_e32_hidden_row_belief.py tests/test_exchange_event_overlay.py tests/test_exchange_event_record.py -q > logs/e33/tests.log 2>&1
$PY -m scripts.run_e33 > logs/e33/runner.log 2>&1 < /dev/null
