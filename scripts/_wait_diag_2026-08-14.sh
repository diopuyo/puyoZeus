#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
LOG=logs/_diag_chain_anim_duration_by_n_2026-08-14.log
PID=151867
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo PROC_ENDED
    break
  fi
  if [ -s "$LOG" ]; then
    echo LOG_HAS_CONTENT
    break
  fi
  sleep 10
done
echo '--- status ---'
ps -o pid,etimes,pcpu,cmd -p "$PID" 2>/dev/null
echo '--- log tail ---'
tail -c 3000 "$LOG"
