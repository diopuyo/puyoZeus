#!/bin/bash
set -e
FFMPEG="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
MP4="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/verify/zenchi_probe_2026-08-21/regionC_6733_7033.mp4"
OUT="/mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/85af3971-d05b-42eb-8a0a-fce931916160/scratchpad/c2_c5_review"
mkdir -p "$OUT"

# C-2: t=6788.5s -> region offset=55.5s (region start=6733.0s)
"$FFMPEG" -y -ss 51.5 -i "$MP4" -t 14 -vf fps=2 "$OUT/C2_%03d.png"
echo "--- C2 done ---"

# C-5: t=6931.7s -> region offset=198.7s
"$FFMPEG" -y -ss 194.7 -i "$MP4" -t 14 -vf fps=2 "$OUT/C5_%03d.png"
echo "--- C5 done ---"

ls -la "$OUT"
