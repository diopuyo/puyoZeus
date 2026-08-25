#!/bin/bash
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
FF=$(./venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
echo "FFMPEG=$FF"
$FF -i data/frames/video_zenchi_c0BQoMJwwQU.mp4 2>&1 | tail -20
