#!/bin/bash
LOG=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/logs/scan_c96_2026-08-08.log
for i in $(seq 1 24); do
  if grep -q "progress" "$LOG" 2>/dev/null; then
    echo "[found progress after ${i}x5s]"
    break
  fi
  sleep 5
done
echo "--- tail ---"
tail -10 "$LOG"
echo "--- proc check ---"
pgrep -fa scan_game_screens
