#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
"$FF" -i data/verify/youtube_demo_2026-08-07/_smoke_show_states.mp4 2>&1 | grep Duration
