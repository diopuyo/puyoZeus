#!/bin/bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(PYTHONPATH=. ./venv/bin/python -c "from src.video_compositer import VideoCompositor; print(VideoCompositor._resolve_ffmpeg_bin())")
DIR=data/verify/advantage_videos_olRyxDGacbg_2026-08-03
for f in match_01 match_02 match_03 match_04 match_05; do
  echo "=== ${f} ==="
  "${FF}" -i "${DIR}/${f}.mp4" 2>&1 | grep -E "Duration|Stream #0"
done
