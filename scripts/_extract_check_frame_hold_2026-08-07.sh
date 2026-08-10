#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
OUT=data/verify/youtube_demo_2026-08-07
"$FF" -y -ss 39.2 -i "$OUT/_smoke_hold.mp4" -frames:v 1 "$OUT/_smoke_hold_t39_2.png"
ls -lh "$OUT/_smoke_hold_t39_2.png"
