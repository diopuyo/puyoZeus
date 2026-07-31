#!/bin/bash
# collect プロセスの動画 fd 読み込みオフセットから概算進捗を出す
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
for pid in $(pgrep -f "venv/bin/python -m scripts.collect_indicators_v2"); do
  args=$(tr '\0' ' ' < /proc/$pid/cmdline)
  video=$(echo "$args" | grep -oE '(data/frames|/home/[a-z]+/frames)/video_[0-9]+\.mp4' | head -1)
  outname=$(echo "$args" | grep -o 'study/[a-z0-9_]*\.csv' | head -1)
  [ -z "$video" ] && continue
  size=$(stat -c %s "$video")
  # 動画 mp4 を開いている fd を探す
  for fd in /proc/$pid/fd/*; do
    tgt=$(readlink "$fd" 2>/dev/null)
    case "$tgt" in
      *video_*.mp4)
        pos=$(grep pos "/proc/$pid/fdinfo/$(basename $fd)" | head -1 | grep -o '[0-9]*')
        pct=$((pos * 100 / size))
        etime=$(ps -o etime= -p $pid | tr -d ' ')
        echo "$outname pid=$pid etime=$etime read=${pct}%"
        break
        ;;
    esac
  done
done
