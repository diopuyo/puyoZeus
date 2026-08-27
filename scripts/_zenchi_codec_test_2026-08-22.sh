#!/bin/bash
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
FF=venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
OUT=data/verify/_codec_test_2026-08-22
mkdir -p "$OUT"

echo "=== 元clip切り出し (120秒、t=1000) ==="
time "$FF" -y -ss 1000 -i data/verify/zenchi_delivery_2026-08-21/zenchi_set1_audio.mp4 -t 120 -c copy "$OUT/src_clip.mp4" 2>&1 | tail -5
ls -la "$OUT/src_clip.mp4"

echo "=== h264 crf20 ==="
time "$FF" -y -i "$OUT/src_clip.mp4" -c:v libx264 -crf 20 -preset medium -c:a aac -b:a 160k "$OUT/h264_crf20.mp4" 2>&1 | tail -15
ls -la "$OUT/h264_crf20.mp4"

echo "=== h264 crf23 ==="
time "$FF" -y -i "$OUT/src_clip.mp4" -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 160k "$OUT/h264_crf23.mp4" 2>&1 | tail -5
ls -la "$OUT/h264_crf23.mp4"

echo "=== 元clip (mpeg4) サイズ ==="
ls -la "$OUT/src_clip.mp4"
