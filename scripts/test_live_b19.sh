#!/usr/bin/env bash
# 一括試験は既定除外のslowも含め、実行条件と結果を保存する。
set -u
cd /mnt/c/Users/ryouj/.codex/worktrees/rt/puyo_analyzer
mkdir -p logs/live_b19
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
PYTHON=${PUYO_TEST_PYTHON:-/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python}
nice -n 19 "$PYTHON" -m pytest tests/ -q -ra -o addopts= \
  --junitxml=logs/live_b19/after.xml > logs/live_b19/after.log 2>&1 < /dev/null
echo "$?" > logs/live_b19/after.exit
