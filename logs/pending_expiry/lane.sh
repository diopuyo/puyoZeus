#!/bin/bash
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for s in "$@"; do
  nice -n 10 $PY -B -m scripts.measure_pending_expiry run $s > logs/pending_expiry/run_$s.log 2>&1
done
