#!/bin/bash
# ウォームアップ版viz2本から対比フレーム抽出 (source秒-2592=クリップ相対秒)
# 313=source2905 (コールドスタート版viz_t30s相当) / 347=source2939 (バースト、viz_t64s相当) / 328=中間
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
OUT=data/verify/youtube_demo_2026-08-07/stills
for t in 313 328 347; do
  nice -n 19 ./venv/bin/ffmpeg -y -loglevel error -ss "$t" \
    -i data/verify/youtube_demo_2026-08-07/dio_vs_ts_full_overlay_viz.mp4 \
    -frames:v 1 "$OUT/warm_full_t${t}s.png"
  nice -n 19 ./venv/bin/ffmpeg -y -loglevel error -ss "$t" \
    -i data/verify/youtube_demo_2026-08-07/dio_vs_ts_stable_only_viz.mp4 \
    -frames:v 1 "$OUT/warm_stableonly_t${t}s.png"
done
ls "$OUT" | grep warm_ | wc -l
