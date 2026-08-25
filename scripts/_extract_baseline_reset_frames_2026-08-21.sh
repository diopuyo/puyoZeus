#!/bin/bash
set -e
FFMPEG="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
MP4="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/verify/zenchi_probe_2026-08-21/regionC_6733_7033.mp4"
OUT="/mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/85af3971-d05b-42eb-8a0a-fce931916160/scratchpad/c2_c5_review"
mkdir -p "$OUT"

# baseline-reset #1: frame=409878 (60fps元動画) -> t=6831.3s -> region offset=98.3s (直前3秒から)
"$FFMPEG" -y -ss 96.3 -i "$MP4" -t 6 -vf fps=2 "$OUT/BR1_%03d.png"
echo "--- BR1 done ---"

# baseline-reset #2: frame=418304 -> t=6971.73s -> region offset=238.73s
"$FFMPEG" -y -ss 236.7 -i "$MP4" -t 6 -vf fps=2 "$OUT/BR2_%03d.png"
echo "--- BR2 done ---"

ls "$OUT" | grep BR
