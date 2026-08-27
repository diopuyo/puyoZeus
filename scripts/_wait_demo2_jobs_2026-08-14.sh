#!/bin/bash
# demo2レンダ(v7、fresh restart) + demo2v2 selfverify(元祖205891) の完了待ち。
# MSYSの \| エスケープ事故 (memory feedback_msys_pipe_escape) を踏まえ、
# pgrep のアルタネーション(|)は使わず単発呼び出しを複数回に分ける。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
i=0
while true; do
  R1=done
  pgrep -f "demo2_video74_3match.mp4" > /dev/null && R1=run
  R2=done
  pgrep -f "_selfverify_demo2v2" > /dev/null && R2=run
  echo "[poll $i] demo2_render=$R1 demo2_sv=$R2 $(date +%H:%M:%S)"
  if [ "$R1" = "done" ] && [ "$R2" = "done" ]; then
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
