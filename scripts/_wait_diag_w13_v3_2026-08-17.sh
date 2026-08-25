#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
LOG=logs/_diag_w13_sideeffect_mechanism_2026-08-17_v3.log
for i in $(seq 1 100); do
  if grep -q "サマリ保存" "$LOG" 2>/dev/null; then
    echo DONE
    break
  fi
  sleep 5
done
tail -c 4000 "$LOG"
