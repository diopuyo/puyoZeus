#!/bin/bash
# レビュー動画をアップロード上限(30MiB)以下に再圧縮
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
D=data/verify/review_video_final_2026-07-26
"$FF" -y -loglevel error -i "$D/advantage_recog_video73_match1_full_score0to0_h264.mp4" \
  -c:v libx264 -preset medium -crf 28 -pix_fmt yuv420p -movflags +faststart \
  "$D/advantage_recog_video73_match1_full_score0to0_h264_small.mp4"
ls -la "$D"/*small*
