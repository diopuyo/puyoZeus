#!/bin/bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
"$FF" -loglevel error -ss 455.0 -i data/frames/video_c34.mp4 -frames:v 1 -y data/verify/review_video_new_2026-07-25/boundary_check/src_455.0s_game0.png
echo ok
