#!/bin/bash
# v4レビュー動画の序盤フレーム抽出(2P青2個の突合用)
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
OUT=data/verify/review_video_new_2026-07-25/v6_early_check
mkdir -p "$OUT"
V=data/verify/review_video_new_2026-07-25/advantage_recog_c34_game1_full_score0to0_v6_h264.mp4
for t in 3.0 4.5 6.0 7.5 9.0 11.0; do
  "$FF" -loglevel error -ss "$t" -i "$V" -frames:v 1 -y "$OUT/v6_${t}s.png"
done
ls "$OUT"
