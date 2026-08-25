#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
echo "ffmpeg: $FF"
"$FF" -i data/verify/demo_fixed_2026-08-13/demo2_video74_3match.mp4 2>&1 | tail -20
