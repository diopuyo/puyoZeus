#!/bin/bash
# 最終版vizの検収フレーム抽出
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
OUT=data/verify/youtube_demo_2026-08-07/stills
for t in 35 39.2 64; do
  nice -n 19 ./venv/bin/ffmpeg -y -loglevel error -ss "$t" \
    -i data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL4a_all_states_viz.mp4 \
    -frames:v 1 "$OUT/final4a_t${t}s.png"
  nice -n 19 ./venv/bin/ffmpeg -y -loglevel error -ss "$t" \
    -i data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL4b_stable_tsumo_viz.mp4 \
    -frames:v 1 "$OUT/final4b_t${t}s.png"
done
ls "$OUT" | grep final4 | wc -l
