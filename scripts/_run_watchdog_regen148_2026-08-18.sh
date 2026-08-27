#!/bin/bash
# 番人 のランナー。python を直接 setsid -f すると即死するため、
# 148本体と同じく「シェルスクリプトを setsid -f する」方式に揃える。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
mkdir -p logs
LOG="logs/watchdog_regen148_2026-08-18.log"
echo "[run_watchdog] 開始 $(date)" >> "${LOG}"
./venv/bin/python -u scripts/_watchdog_regen148_2026-08-18.py >> "${LOG}" 2>&1
echo "[run_watchdog] 終了 $(date) rc=$?" >> "${LOG}"
