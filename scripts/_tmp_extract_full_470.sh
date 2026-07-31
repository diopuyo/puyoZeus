#!/bin/bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
OUT="/mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/f3efc5f5-b2ab-4019-b80c-3a2d35f86017/scratchpad"
"$FF" -loglevel error -ss 470.1 -i data/frames/video_c34.mp4 -frames:v 1 -y "$OUT/raw_470.1_full.png"
"$FF" -loglevel error -ss 469.0 -i data/frames/video_c34.mp4 -frames:v 1 -y "$OUT/raw_469.0_full.png"
"$FF" -loglevel error -ss 471.0 -i data/frames/video_c34.mp4 -frames:v 1 -y "$OUT/raw_471.0_full.png"
"$FF" -loglevel error -ss 472.0 -i data/frames/video_c34.mp4 -frames:v 1 -y "$OUT/raw_472.0_full.png"
ls -la "$OUT"
