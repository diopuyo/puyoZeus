#!/bin/bash
# 停滞検知の番人 (2026-08-19、41本収集+DL2系統を監視)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
mkdir -p logs
exec >> logs/stall_watchdog_2026-08-19.log 2>&1
echo "[run_stall] 開始 $(date)"
./venv/bin/python -u scripts/_stall_watchdog_2026-08-18.py   --config scripts/_stall_watchdog_targets_2026-08-19.json
echo "[run_stall] 終了 $(date) rc=$?"
