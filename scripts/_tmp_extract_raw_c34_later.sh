#!/bin/bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
mkdir -p /tmp/c34raw
for t in 478.0 480.0 483.0 486.0 490.0; do
  "$FF" -loglevel error -ss "$t" -i data/frames/video_c34.mp4 -frames:v 1 -y "/tmp/c34raw/raw_${t}.png"
done
ls -la /tmp/c34raw
