#!/bin/bash
# 本番5試合デモ用: olRyxDGacbg (A級 DIO vs TS 30先) を1080pで全体DL。
# --download-sections はネットワークdownloader経由のffmpeg呼び出しで
# exit -11 (segfault) する既知の罠のため使わない (全体DL→ローカルtrim方式)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF="$(pwd)/venv/bin"
OUT="data/frames/video_olRyxDGacbg.mp4"
if [ -s "$OUT" ]; then echo "[skip] already exists: $OUT"; ls -la "$OUT"; exit 0; fi
rm -f "$OUT" "$OUT".part
echo "[start] $(date)"
./venv/bin/python -m yt_dlp --ffmpeg-location "$FF" \
  -f 'bv*[vcodec^=avc1][height<=1080]+ba/b[ext=mp4][vcodec^=avc1][height<=1080]/b[height<=1080][vcodec!*=av01]/b[ext=mp4]' \
  --remux-video mp4 --no-playlist --no-progress -o "$OUT" \
  "https://www.youtube.com/watch?v=olRyxDGacbg"
echo "[exit=$?] $(date)"
ls -la "$OUT"
