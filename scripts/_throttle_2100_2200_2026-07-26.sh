#!/bin/bash
# 2026-07-26 user指示: 21:00-22:00 は CPU/メモリを25%以上空ける。
# 走行中の重いPythonジョブを21:00にSIGSTOPで凍結し、22:00にSIGCONTで再開する。
set -u
LOG=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/logs/throttle_2100_2200_2026-07-26.log
PATTERN='measure_stable_cell_acc|_collect_1t|extract_match_winners|_zap_1t'

now_epoch=$(date +%s)
start_epoch=$(date -d "21:00" +%s)
end_epoch=$(date -d "22:00" +%s)

if [ "$now_epoch" -lt "$start_epoch" ]; then
  sleep $((start_epoch - now_epoch))
fi
echo "[throttle] STOP at $(date)" >> "$LOG"
pkill -STOP -f "$PATTERN" 2>>"$LOG" || true
pgrep -f measure_stable_cell_acc >> "$LOG" 2>&1

now_epoch=$(date +%s)
if [ "$now_epoch" -lt "$end_epoch" ]; then
  sleep $((end_epoch - now_epoch))
fi
echo "[throttle] CONT at $(date)" >> "$LOG"
pkill -CONT -f "$PATTERN" 2>>"$LOG" || true
echo "[throttle] done at $(date)" >> "$LOG"
