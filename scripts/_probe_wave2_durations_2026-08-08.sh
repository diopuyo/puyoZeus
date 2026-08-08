#!/bin/bash
# wave2未着手分の動画長を確認 (90分超の放送型を特定)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=./venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
for n in 96 97 98 99 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144; do
  f="data/frames/video_c${n}.mp4"
  [ -f "$f" ] || { echo "c${n}: MISSING"; continue; }
  dur=$($FF -i "$f" 2>&1 | grep Duration | head -1 | grep -oE "[0-9]{2}:[0-9]{2}:[0-9]{2}")
  echo "c${n}: ${dur}"
done
