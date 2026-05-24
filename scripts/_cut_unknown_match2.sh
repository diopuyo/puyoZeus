#!/usr/bin/env bash
# unknown video から match 2 (= 41s から 75s) を切り出す
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
FFMPEG="venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
"$FFMPEG" -y -i data/test_unknown/unknown_match_120s.mp4 -ss 41 -t 75 -c copy data/test_unknown/unknown_match2_75s.mp4
echo "done"
