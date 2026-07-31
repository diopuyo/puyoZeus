#!/bin/bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
mkdir -p /tmp/c34raw
"$FF" -loglevel error -ss 470.1 -i data/frames/video_c34.mp4 -frames:v 1 -y /tmp/c34raw/raw_470.1.png
"$FF" -loglevel error -ss 473.1 -i data/frames/video_c34.mp4 -frames:v 1 -y /tmp/c34raw/raw_473.1.png
"$FF" -loglevel error -ss 468.8 -i data/frames/video_c34.mp4 -frames:v 1 -y /tmp/c34raw/raw_468.8.png
ls -la /tmp/c34raw
