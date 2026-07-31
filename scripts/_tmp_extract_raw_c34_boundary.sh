#!/bin/bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
mkdir -p /tmp/c34raw
for t in 466.0 468.0 469.0 470.5 471.0 471.5 472.0 474.0 476.0; do
  "$FF" -loglevel error -ss "$t" -i data/frames/video_c34.mp4 -frames:v 1 -y "/tmp/c34raw/raw_${t}.png"
done
