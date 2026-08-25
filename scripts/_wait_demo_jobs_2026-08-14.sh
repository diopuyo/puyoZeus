#!/bin/bash
# 検収デモ再生成4ジョブ (確認デモrender/demo2render/確認デモselfverify/demo2 selfverify)
# の完了をポーリングする (MSYSエスケープ事故回避のためスクリプトファイル化)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
i=0
while true; do
  R1=done
  pgrep -f "demo_fixed_3match.mp4" > /dev/null && R1=run
  R2=done
  pgrep -f "demo2_video74_3match.mp4" > /dev/null && R2=run
  R3=done
  pgrep -f "_selfverify_final2_confirm" > /dev/null && R3=run
  R4=done
  pgrep -f "_selfverify_demo2v2" > /dev/null && R4=run
  echo "[poll $i] confirm_render=$R1 demo2_render=$R2 confirm_sv=$R3 demo2_sv=$R4 $(date +%H:%M:%S)"
  if [ "$R1" = "done" ] && [ "$R2" = "done" ] && [ "$R3" = "done" ] && [ "$R4" = "done" ]; then
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
