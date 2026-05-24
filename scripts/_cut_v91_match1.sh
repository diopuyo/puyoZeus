#!/usr/bin/env bash
# v91 から match 1 (195-270s) を切り出して 720p unknown 試験動画にする
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
FFMPEG="venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
mkdir -p data/test_unknown
"$FFMPEG" -y -ss 195 -i data/frames/video_91.mp4 -t 75 -c copy data/test_unknown/v91_match1_75s_720p.mp4
echo "done"
