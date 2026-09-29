#!/usr/bin/env bash
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/rt/puyo_analyzer
mkdir -p logs/live_b18c
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
exec > logs/live_b18c/final-video.log 2>&1 < /dev/null
for source in review zenchi; do
  nice -n 10 "$PYTHON" -u -m scripts.verify_live_b18c --source "$source" --output-root logs/live_b18c/final
done
