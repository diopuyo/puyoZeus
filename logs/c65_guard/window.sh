#!/bin/bash
# q第14試合の窓 (852〜886秒) を実フラグ OFF/ON で順に再生する
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for m in off on; do
  nice -n 10 $PY -B scripts/measure_c65_guard.py window 852 886 $m > logs/c65_guard/window_$m.log 2>&1
done
