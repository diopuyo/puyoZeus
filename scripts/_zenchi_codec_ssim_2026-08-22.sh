#!/bin/bash
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
FF=venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
OUT=data/verify/_codec_test_2026-08-22

echo "=== SSIM/PSNR: h264 crf20 vs 元mpeg4 ==="
"$FF" -i "$OUT/h264_crf20.mp4" -i "$OUT/src_clip.mp4" -lavfi "ssim;[0:v][1:v]psnr" -f null - 2>&1 | grep -E "SSIM|PSNR"

echo "=== SSIM/PSNR: h264 crf23 vs 元mpeg4 ==="
"$FF" -i "$OUT/h264_crf23.mp4" -i "$OUT/src_clip.mp4" -lavfi "ssim;[0:v][1:v]psnr" -f null - 2>&1 | grep -E "SSIM|PSNR"
