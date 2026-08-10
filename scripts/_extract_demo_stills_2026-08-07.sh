#!/bin/bash
# YouTubeデモ用: 8連鎖バースト帯の候補フレーム抽出 (clip相対秒)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
CLIP=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
OUT=data/verify/youtube_demo_2026-08-07/stills
mkdir -p "$OUT"
for t in 58 60 62 63 64 65 66 68 70 74; do
  nice -n 19 ./venv/bin/ffmpeg -y -loglevel error -ss "$t" -i "$CLIP" \
    -frames:v 1 "$OUT/clip_t${t}s.png"
done
ls "$OUT" | wc -l
