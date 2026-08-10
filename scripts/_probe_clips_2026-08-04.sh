#!/bin/bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF="$(PYTHONPATH=. ./venv/bin/python -c "from src.video_compositer import VideoCompositor; print(VideoCompositor._resolve_ffmpeg_bin())")"
for f in data/verify/advantage_videos_olRyxDGacbg_2026-08-03/clip_*_final.mp4; do
  echo "== ${f} =="
  "${FF}" -i "${f}" 2>&1 | grep -E "Duration|Stream #0:0"
done
