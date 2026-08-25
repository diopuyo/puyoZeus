#!/bin/bash
# 先回りDL のランナー。python を直接 setsid -f すると即死するため、
# 148本体と同じく「シェルスクリプトを setsid -f する」方式に揃える
# (2026-08-18: 直接起動でログ0バイトのまま起動しない事象を実測)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
mkdir -p logs
LOG="logs/prefetch_dl_failures_2026-08-18.log"
echo "[run_prefetch] 開始 $(date)" >> "${LOG}"
./venv/bin/python -u scripts/_prefetch_dl_failures_2026-08-18.py >> "${LOG}" 2>&1
echo "[run_prefetch] 終了 $(date) rc=$?" >> "${LOG}"
