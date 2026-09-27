#!/usr/bin/env bash
set -u
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. PYTHONUNBUFFERED=1 PYTHONHASHSEED=20260926
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
"$PYTHON" -m scripts.run_e13_exchange_replay_20260927 > logs/e13/final_replay.log 2>&1 &
REPLAY_PID=$!
while [ ! -f logs/e13/sha.exit ] || [ "$(cat logs/e13/sha.exit)" != 0 ]; do sleep 5; done
bash scripts/_render_e13_20260927.sh
echo $? > logs/e13/render.exit
wait "$REPLAY_PID"
echo $? > logs/e13/final_replay.exit
"$PYTHON" -m scripts.report_e13_20260927 > logs/e13/report.log 2>&1
echo $? > logs/e13/report.exit
