#!/bin/bash
# 3412.2s~4379.5s の間に大きな不連続(WIN星の急変・2P名変化)があるか粗く走査
set -e
cd "/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
SRC="data/frames/video_zenchi_c0BQoMJwwQU.mp4"
OUTDIR="data/verify/zenchi_boundary_confirm_2026-08-21/gapscan"
mkdir -p "$OUTDIR"

extract() {
  t="$1"; name="$2"
  pre=$(awk -v t="$t" 'BEGIN{v=t-3; if(v<0)v=0; printf "%.3f", v}')
  off=$(awk -v t="$t" -v pre="$pre" 'BEGIN{printf "%.3f", t-pre}')
  ffmpeg -y -ss "$pre" -i "$SRC" -ss "$off" -frames:v 1 -q:v 2 "$OUTDIR/$name" >/dev/null 2>&1
}

for t in 3430 3460 3500 3550 3600 3650 3700 3750 3800 3850 3900 3950 4000 4050 4100 4150 4200 4250 4300 4350; do
  extract "$t" "gap_t${t}.png"
done
echo "done: $(ls "$OUTDIR" | wc -l) files"
