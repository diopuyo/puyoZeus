#!/bin/bash
# 1レーン: 引数の動画を順に影子測定する
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for s in "$@"; do
  nice -n 10 $PY -B scripts/measure_c65_guard.py run $s > logs/c65_guard/run_$s.log 2>&1
done
