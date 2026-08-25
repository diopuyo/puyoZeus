#!/usr/bin/env bash
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
D=data/verify/youtube_demo_2026-08-07/release
FF=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
"$FF" -nostdin -hide_banner -loglevel error -y -i "$D/final_3_advantage_only.mp4"   -vf scale=-2:720 -c:v libx264 -preset veryfast -crf 28 -pix_fmt yuv420p   -movflags +faststart -an "$D/mobile/final_3_advantage_only_720p.mp4"
ls -la --time-style=+%H:%M "$D/mobile/"
