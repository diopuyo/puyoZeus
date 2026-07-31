#!/bin/bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
OUT_DIR="data/verify/strict_fix_verify_2026-07-26"
for tt in 0.5 1.0 2.0 3.0 5.0 8.0 10.0; do
  "${FF}" -y -ss "${tt}" -i "${OUT_DIR}/c34_v7_strict_check.mp4" \
    -frames:v 1 "${OUT_DIR}/c34_v7_frame_t${tt}.png" -loglevel error
done
ls -la "${OUT_DIR}"/c34_v7_frame_t*.png
