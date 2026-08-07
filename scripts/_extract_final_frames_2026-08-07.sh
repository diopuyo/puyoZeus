#!/bin/bash
# 最終版vizの検収フレーム抽出
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
OUT=data/verify/youtube_demo_2026-08-07/stills
for t in 32 33.5 49 52; do
  nice -n 19 ./venv/bin/ffmpeg -y -loglevel error -ss "$t" \
    -i data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL3_v3_hold_viz.mp4 \
    -frames:v 1 "$OUT/final3_t${t}s.png"
done
ls "$OUT" | grep final3_ | wc -l
