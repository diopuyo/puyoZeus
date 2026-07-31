#!/bin/bash
# combined30_video_breakdown 完了待ちポーリング (単純パターン、パイプ文字不使用)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
while true; do
  n=$(pgrep -c -f _tmp_video_phase_auc_breakdown)
  if [ "$n" -eq 0 ]; then
    echo "BREAKDOWN_DONE"
    break
  fi
  sleep 5
done
