#!/bin/bash
# 指摘1(c56_g3 51秒)・指摘2(c65_g3 1分)検証用フレーム抽出。
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
FF=./venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
OUTDIR=/tmp/frames_review4
mkdir -p "$OUTDIR"

for t in 330 335 339 341 343 345 350 355 360; do
  "$FF" -y -loglevel error -ss "$t" -i data/frames/video_c56.mp4 -frames:v 1 "$OUTDIR/c56_t${t}.png"
done
for t in 356 360 364 366 368 370 375 380 385; do
  "$FF" -y -loglevel error -ss "$t" -i data/frames/video_c65.mp4 -frames:v 1 "$OUTDIR/c65_t${t}.png"
done
ls -la "$OUTDIR"
