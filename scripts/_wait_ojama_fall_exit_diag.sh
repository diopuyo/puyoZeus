#!/usr/bin/env bash
# 一時 wait スクリプト (scratchpad的用途、診断run完了待ち専用)。
LOG="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/logs/diag_ojama_fall_exit_timing_2026-07-24.log"
while ! grep -q "\[DONE\]" "$LOG" 2>/dev/null; do
  sleep 20
done
echo "===== DONE detected ====="
tail -120 "$LOG"
