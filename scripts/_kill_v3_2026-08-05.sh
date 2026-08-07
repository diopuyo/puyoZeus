#!/bin/bash
# v3走行の完全停止 (2026-08-05、使い捨て)。ランナー→子の順で確実に殺す
runner=$(pgrep -f "_run_burst_guard_v3_2026-08-05.sh")
[ -n "$runner" ] && echo "runner: $runner" && kill -9 $runner
sleep 1
kids=$(pgrep -f "burst_guard_2026-08-05/on_v3/")
[ -n "$kids" ] && echo "kids: $kids" && kill -9 $kids
sleep 2
echo "残存: $(pgrep -c -f collect_boards_lean || echo 0)"
