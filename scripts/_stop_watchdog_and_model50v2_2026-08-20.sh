#!/bin/bash
# 番人と48本収集を止め、A/B 検証収集 (zeroreset) だけを残す (2026-08-20)。
# 番人は停止フラグ (logs/.stall_watchdog_paused) を立てても 48本収集を
# 再起動したため、番人プロセス自体を止める。
# 複雑なコマンド置換は wsl 経由の直書きだと MSYS に壊されるためスクリプト化
# (memory feedback_msys_pipe_escape)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

pkill -f _run_stall_watchdog
pkill -f _stall_watchdog_2026-08-18
sleep 3
pkill -f _regen_model50v2_2026-08-20
pkill -f _run_model50v2_2026-08-20
sleep 3
pkill -9 -f boards_lean_model50v2_2026-08-20
sleep 4

echo -n "番人プロセス: "
pgrep -c -f stall_watchdog || echo 0
echo -n "model50v2 収集: "
pgrep -c -f boards_lean_model50v2 || echo 0
echo -n "zeroreset A/B 収集: "
pgrep -c -f boards_lean_zeroreset || echo 0
uptime
