#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
i=0
while true; do
  R=done
  pgrep -f "demo2_video74_3match.mp4" > /dev/null && R=run
  echo "[poll $i] demo2_render=$R $(date +%H:%M:%S)"
  if [ "$R" = "done" ]; then
    echo ALL_DONE
    break
  fi
  i=$((i + 1))
  if [ "$i" -ge 27 ]; then
    echo TIMED_OUT_STILL_RUNNING
    break
  fi
  sleep 20
done
