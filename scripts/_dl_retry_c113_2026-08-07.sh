#!/bin/bash
# c113 (403) を更新済みyt-dlpで単独再試行
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin
out="data/frames/video_c113.mp4"
rm -f data/frames/video_c113.*.part "$out"
nice -n 15 ./venv/bin/python -m yt_dlp --no-update --ffmpeg-location "$FF" \
  -f 'bv*[vcodec^=avc1][height<=1080]+ba/b[ext=mp4][vcodec^=avc1][height<=1080]/b[height<=1080][vcodec!*=av01]/b[ext=mp4]' \
  --remux-video mp4 --no-playlist --no-progress -o "$out" \
  "https://www.youtube.com/watch?v=kOWy50IddfI" 2>&1 | tail -5
if [ -s "$out" ]; then echo "[OK] video_c113"; else echo "[STILL-FAIL] video_c113"; fi
