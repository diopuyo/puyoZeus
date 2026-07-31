#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
echo "FF=$FF"
"$FF" -filters 2>&1 | grep -i drawtext
"$FF" -version 2>&1 | head -3
ls -la /mnt/c/Windows/Fonts/meiryo.ttc
