#!/bin/bash
# native 特徴の出力一致確認を detach 起動する。使い方: launch_parity.sh <source>
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
mkdir -p logs/prefire_prediction/native_parity
setsid -f bash -c "$PY -B -m scripts.prefire_native_feature_parity_20261001 --source $1 > logs/prefire_prediction/native_parity/$1.log 2>&1 < /dev/null"
