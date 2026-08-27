#!/bin/bash
# 一時ポーリングスクリプト: subset42のCSVビルド完了待ち (2026-08-19)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
OUT_CSV="data/verify/labeled_win_subset42_2026-08-19/labeled.csv"
PID=81508
for i in $(seq 1 60); do
  if [ -f "$OUT_CSV" ]; then
    echo "FOUND"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "PROC_GONE"
    exit 0
  fi
  echo "tick_$i"
  sleep 10
done
echo "TIMEOUT"
