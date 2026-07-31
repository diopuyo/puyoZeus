#!/bin/bash
# 既存レンダ済みオーバーレイ動画から表示値(勝率バー等)を確認するためのフレーム抽出。
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
FF=./venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
OUTDIR=/tmp/overlay_frames
mkdir -p "$OUTDIR"

# c56_g3: clip相対 (start-sec=288) -> 51秒付近を相対秒で
for t in 40 45 49 51 53 55 57 60 65 70; do
  "$FF" -y -loglevel error -ss "$t" -i data/verify/review4_2026-07-29/advantage_c56_g3_full_score0to0_h264.mp4 -frames:v 1 "$OUTDIR/ov_c56_t${t}.png"
done
# c65_g3: clip相対 (start-sec=306) -> 1分付近(58-72s)を相対秒で
for t in 50 55 58 60 62 65 68 70 75 80; do
  "$FF" -y -loglevel error -ss "$t" -i data/verify/review4_2026-07-29/advantage_c65_g3_full_score0to0_h264.mp4 -frames:v 1 "$OUTDIR/ov_c65_t${t}.png"
done
ls -la "$OUTDIR"
