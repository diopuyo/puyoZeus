#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
mkdir -p logs/e16/on
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo '窒息信号の補完・学習完了を待機' > logs/e16/on/runner.log
while [ ! -f logs/e16/records/STATUS.json ]; do sleep 15; done
while ! grep -q COMPLETE logs/e16/train/STATUS.json 2>/dev/null; do sleep 15; done
/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -u -m scripts.run_e16_replay_20260928 >> logs/e16/on/runner.log 2>&1 < /dev/null
