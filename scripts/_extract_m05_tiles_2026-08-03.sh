#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
OUT=data/verify/advantage_videos_olRyxDGacbg_2026-08-03/frames
for i in 0 1 2 3 4 5 6 7; do
  t=$(echo "51.0 + $i * 0.5" | bc)
  fname="${OUT}/m05_tile_${i}.png"
  venv/bin/ffmpeg -y -ss "$t" -i data/verify/advantage_videos_olRyxDGacbg_2026-08-03/match_05.mp4 \
    -frames:v 1 -update 1 "$fname" 2>&1 | tail -1
  echo "t=$t -> $fname"
done
