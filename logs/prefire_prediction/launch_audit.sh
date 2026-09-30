#!/bin/bash
# prefire リーク監査 (打ち切り再生) を detach 起動する。使い方: launch_audit.sh <variant> <cut_sec>
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
mkdir -p logs/prefire_prediction/truncation
setsid -f bash -c "$PY -B -m scripts.prefire_truncation_audit_20260930 --variant $1 --cut $2 > logs/prefire_prediction/truncation/$1_$2.log 2>&1 < /dev/null"
