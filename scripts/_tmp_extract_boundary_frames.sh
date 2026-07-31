#!/bin/bash
# c34 game1 境界確認用フレーム抽出 (2026-07-25)
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
OUT=data/verify/review_video_new_2026-07-25/boundary_check
mkdir -p "$OUT"
for t in 465.6 467.0 468.5 469.5 470.5 471.5 472.0 473.0 474.5 476.0; do
  "$FF" -loglevel error -ss "$t" -i data/frames/video_c34.mp4 -frames:v 1 -y "$OUT/src_${t}s.png"
done
ls -la "$OUT"
