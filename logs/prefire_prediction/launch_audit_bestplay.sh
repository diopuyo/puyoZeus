#!/bin/bash
# Phase 3 のリーク監査 (q 打ち切り再生) を detach 起動する。使い方: launch_audit_bestplay.sh <cut_sec> <latency> <tag> <evaluator>
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
mkdir -p logs/prefire_prediction/truncation
setsid -f bash -c "$PY -B -m scripts.prefire_truncation_audit_20260930 --variant bestplay --cut $1 --latency $2 --tag $3 --evaluator $4 > logs/prefire_prediction/truncation/bestplay$3_$1.log 2>&1 < /dev/null"
