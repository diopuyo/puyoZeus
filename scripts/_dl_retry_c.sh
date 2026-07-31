#!/bin/bash
# 403で落ちた c9/c31 を個別再DL(H.264強制)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin
names=(c9 c31)
ids=(mtNoLwHSg1A 80W5JXVxM7U)
for i in "${!names[@]}"; do
  n="${names[$i]}"; id="${ids[$i]}"
  rm -f "data/frames/video_$n".*.part "data/frames/video_$n.mp4"
  echo "[retry] video_$n <- $id"
  nice -n 15 ./venv/bin/python -m yt_dlp --ffmpeg-location "$FF" \
    -f 'bv*[vcodec^=avc1][height<=1080]+ba/b[ext=mp4][vcodec^=avc1][height<=1080]/b[height<=1080][vcodec!*=av01]/b[ext=mp4]' \
    --remux-video mp4 --no-playlist --no-progress -o "data/frames/video_$n.mp4" \
    "https://www.youtube.com/watch?v=$id" 2>&1 | tail -2
done
echo "[retry] done"
ls -lh data/frames/video_c9.mp4 data/frames/video_c31.mp4 2>/dev/null
