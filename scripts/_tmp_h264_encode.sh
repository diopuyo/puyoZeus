#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
IN="data/indicators_v2/overlay/advantage_v29_saturated_check.mp4"
OUT="data/indicators_v2/overlay/advantage_v29_saturated_h264.mp4"
"$FF" -y -i "$IN" -c:v libx264 -preset medium -crf 24 -pix_fmt yuv420p -movflags +faststart "$OUT" 2>/dev/null
ls -la "$OUT"
