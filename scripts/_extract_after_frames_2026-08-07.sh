#!/bin/bash
# 学習後viz対比フレーム抽出 (m01クリップ相対: t=30がBefore viz_t30s相当、t=64がバースト)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
OUT=data/verify/youtube_demo_2026-08-07/stills
for t in 30 64; do
  nice -n 19 ./venv/bin/ffmpeg -y -loglevel error -ss "$t" \
    -i data/verify/youtube_demo_2026-08-07/dio_vs_ts_AFTER_full_viz.mp4 \
    -frames:v 1 "$OUT/after_full_t${t}s.png"
done
nice -n 19 ./venv/bin/ffmpeg -y -loglevel error -ss 30 \
  -i data/verify/youtube_demo_2026-08-07/dio_vs_ts_AFTER_stable_only_viz.mp4 \
  -frames:v 1 "$OUT/after_stableonly_t30s.png"
ls "$OUT" | grep after_ | wc -l
