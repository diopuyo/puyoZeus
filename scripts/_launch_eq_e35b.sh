#!/usr/bin/env bash
set -u
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
for s in q_7gc4TgFig review fcXG83vInDY mia8KCjr52g zenchi; do
  nice -n 10 $PY -B -m scripts._eq_e35b $s > logs/e35b/eq_e36b/$s.log 2>&1 < /dev/null
done
echo ALLDONE > logs/e35b/eq_e36b/ALLDONE
