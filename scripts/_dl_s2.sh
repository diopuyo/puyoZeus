#!/bin/bash
# s2 のみ再DL(ネスト bash -c の変数展開バグ回避のためスクリプト化・絶対パスリテラル)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
rm -f data/frames/video_s2*.part data/frames/video_s2.mp4
nice -n 15 ./venv/bin/python -m yt_dlp \
  --ffmpeg-location /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin \
  -f 'bv*[height<=1080]+ba/b[height<=1080]/b' --remux-video mp4 \
  --no-playlist --no-progress -o 'data/frames/video_s2.%(ext)s' \
  'https://www.youtube.com/watch?v=UpnGj22itdA'
echo "[s2] done"
ls -lh data/frames/video_s2.mp4 2>/dev/null
