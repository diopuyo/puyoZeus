#!/bin/bash
# hazard 学習サンプル作成を nice 0 で detach 起動する。使い方: launch_samples.sh <shard> <shards> [extra args]
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
mkdir -p logs/prefire_prediction/hazard_samples
setsid -f bash -c "nice -n 0 $PY -B -m scripts.prefire_hazard_samples_20260930 --shard $1 --shards $2 $3 $4 > logs/prefire_prediction/hazard_samples/shard_$1_of_$2.log 2>&1 < /dev/null"
