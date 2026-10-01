#!/usr/bin/env bash
# 使い方: _run_switch_replay.sh <variant> <source...>   (1 プロセス・nice 10、順次)
set -eu
cd /mnt/d/puyo_analyzer/wt_switch
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
variant=$1; shift
mkdir -p logs/switch_smoothing/$variant
for s in "$@"; do
  nice -n 10 $PY -B -m scripts.run_switch_smoothing --variant $variant --step replay --source $s \
    >> logs/switch_smoothing/$variant/replay_$s.log 2>&1
done
touch logs/switch_smoothing/$variant/REPLAY_DONE_$(echo "$@" | tr ' ' '_')
