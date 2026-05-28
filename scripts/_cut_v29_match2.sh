#!/bin/bash
# v29 match #2 切り出しスクリプト (197.0s - 275.5s = 78.5s)
set -e

PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
FFMPEG="${PROJ_DIR}/venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
INPUT="${PROJ_DIR}/data/frames/video_29.mp4"
OUTPUT_DIR="${PROJ_DIR}/data/match_clips/v29"
OUTPUT="${OUTPUT_DIR}/match_v29_02.mp4"

echo "[cut] ffmpeg: ${FFMPEG}"
echo "[cut] input: ${INPUT}"
echo "[cut] output: ${OUTPUT}"
echo "[cut] range: 197.0s - 275.5s (78.5s)"

mkdir -p "${OUTPUT_DIR}"

"${FFMPEG}" -ss 197.0 -i "${INPUT}" -t 78.5 \
  -c:v libx264 -preset fast -an \
  -y "${OUTPUT}" 2>&1

echo "[cut] 完了: ${OUTPUT}"
ls -lh "${OUTPUT}"
