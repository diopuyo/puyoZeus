#!/bin/bash
# 新規動画の解像度・ビットレート実測 (デモ動画+新規DL数本+参照用ライブラリ動画)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=./venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
for v in olRyxDGacbg c96 c115 c120 c130 c140 c34 c55; do
  f="data/frames/video_${v}.mp4"
  [ -f "$f" ] || { echo "$v: MISSING"; continue; }
  info=$($FF -i "$f" 2>&1 | grep -E "Stream.*Video" | head -1)
  size=$(stat -c%s "$f")
  dur=$($FF -i "$f" 2>&1 | grep Duration | head -1 | grep -oE "[0-9]{2}:[0-9]{2}:[0-9]{2}")
  echo "$v | $(echo $info | grep -oE '[0-9]{3,4}x[0-9]{3,4}') | $(echo $info | grep -oE '[0-9]+ kb/s' | head -1) | ${dur} | $((size/1024/1024))MB"
done
