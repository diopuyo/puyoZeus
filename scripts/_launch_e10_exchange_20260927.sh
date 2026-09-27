#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. PYTHONUNBUFFERED=1 PYTHONHASHSEED=20260926
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
mkdir -p logs/e10/zenchi
"$PYTHON" -m scripts.replay_exchange_event_20260926 logs/review_zenchi_part3/on_e9/inputs.jsonl.gz --out logs/e10/zenchi > logs/e10/zenchi.log 2>&1
echo 0 > logs/e10/zenchi.exit
