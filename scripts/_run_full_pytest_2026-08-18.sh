#!/bin/bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
mkdir -p logs
exec >> logs/full_pytest_final_2026-08-18.log 2>&1
echo "[pytest] 開始 $(date)"
./venv/bin/python -m pytest tests/ -q -n 8
echo "[pytest] 終了 $(date) rc=$?"
