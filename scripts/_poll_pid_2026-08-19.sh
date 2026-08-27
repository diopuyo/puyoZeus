#!/bin/bash
# 一時ポーリングスクリプト: 指定PIDの終了待ち (2026-08-19、subset42タスク用)
PID="$1"
MAXTICK="${2:-60}"
for i in $(seq 1 "$MAXTICK"); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "PROC_GONE"
    exit 0
  fi
  echo "tick_$i"
  sleep 10
done
echo "TIMEOUT"
