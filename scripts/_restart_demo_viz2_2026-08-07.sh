#!/bin/bash
# デモviz2本のクリーン再起動 (既存viz残骸をkill→部分出力削除→detach起動→生存確認)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
pkill -f "scripts.visualize_recognition" 2>/dev/null
sleep 3
rm -f data/verify/youtube_demo_2026-08-07/dio_vs_ts_full_overlay_viz.mp4 \
      data/verify/youtube_demo_2026-08-07/dio_vs_ts_stable_only_viz.mp4
: > logs/demo_viz_full_2026-08-07.log
: > logs/demo_viz_stable_only_2026-08-07.log
setsid -f bash scripts/_gen_demo_viz2_2026-08-07.sh > logs/demo_viz2_runner_2026-08-07.log 2>&1 < /dev/null
sleep 10
echo "alive_viz=$(pgrep -cf 'scripts.visualize_recognition')"
