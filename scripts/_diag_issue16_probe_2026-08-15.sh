#!/bin/bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
FF=$(./venv/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
echo "FF=$FF"
"$FF" -i data/frames/review_demo_2026-08-12.mp4 > logs/_diag_issue16_ffprobe_2026-08-15.log 2>&1
cat logs/_diag_issue16_ffprobe_2026-08-15.log
