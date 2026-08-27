#!/bin/bash
# v35レビュー動画の映像・音声を全編デコードする。
set -euo pipefail
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

VIDEO=data/verify/gate4_first5_review_2026-08-27/gate4_first5_cond5_v35_review_extreme_flip_guard.mp4
FFMPEG=$(./venv/bin/python -c \
  'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
"$FFMPEG" -v error -i "$VIDEO" -map 0:v:0 -map 0:a:0 -f null -
echo FULL_AV_DECODE_PASS
