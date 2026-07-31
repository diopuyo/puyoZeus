#!/bin/bash
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
FF=./venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
OUTDIR=/tmp/overlay_frames2
mkdir -p "$OUTDIR"
for t in 71 72 73 74; do
  "$FF" -y -loglevel error -ss "$t" -i data/verify/review4_2026-07-29/advantage_c56_g3_full_score0to0_h264.mp4 -frames:v 1 "$OUTDIR/ov_c56_t${t}.png"
done
ls -la "$OUTDIR"
