#!/bin/bash
# 再生ワーカーが最大2本になった時点で実行する。
set -eu
cd /mnt/d/puyo_analyzer/wt_margin
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
nice -n 10 "$PY" -B -m scripts.audit_margin_origins > logs/margin_clock/origin_audit.log
nice -n 10 "$PY" -B -m pytest -q tests/test_margin_clock.py tests/test_exchange_event_overlay.py tests/test_exchange_event_production.py tests/test_ojama_accounting.py tests/test_e12_exchange_evaluation.py tests/test_e10b_exchange_landing.py tests/test_e10c_exchange_landing.py tests/test_e4_exchange_refresh.py tests/test_exchange_event_record.py tests/test_exchange_event_tracker.py tests/test_exchange_event_evaluator.py tests/test_e16_layers.py tests/test_e16_sync.py > logs/margin_clock/tests.log 2>&1
