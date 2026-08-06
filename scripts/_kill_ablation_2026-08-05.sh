#!/bin/bash
# ablation走行 (tguard/thr954) の完全停止 (2026-08-05、使い捨て)
# ランナー (bash) を先に殺してから子 (collect_boards_lean) を殺す
for pat in "burst_guard_factorial" "on_v2_tguard" "on_v2_thr954"; do
  pids=$(pgrep -f "$pat")
  if [ -n "$pids" ]; then
    echo "kill: $pat -> $pids"
    kill -9 $pids 2>/dev/null
  fi
done
sleep 2
echo "残存 collect: $(pgrep -c -f collect_boards_lean || echo 0)"
