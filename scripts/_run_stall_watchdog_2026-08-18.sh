#!/bin/bash
# 汎用 停滞・死亡検知 番人 のランナー。
# python を直接 setsid -f すると起動しないことがある (2026-08-18 実測、
# 番人/先回りDL/148本体の3件で発生)。148本体と同じく「シェルスクリプトを
# setsid -f bash する」方式に揃え、さらに exec でリダイレクトを固定することで
# 外部リダイレクトのバッファ滞留も避ける。
#
# 使い方 (WSL detach、長時間放置前提、CLAUDE.md プロセス管理ルール):
#   wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \
#     setsid -f bash scripts/_run_stall_watchdog_2026-08-18.sh < /dev/null"
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
mkdir -p logs
exec >> logs/stall_watchdog_2026-08-18.log 2>&1
echo "[run_stall_watchdog] 開始 $(date)"
./venv/bin/python -u scripts/_stall_watchdog_2026-08-18.py
echo "[run_stall_watchdog] 終了 $(date) rc=$?"
