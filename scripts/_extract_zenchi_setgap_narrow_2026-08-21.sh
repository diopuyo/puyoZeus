#!/bin/bash
# 3480~3700s を5秒刻みで走査し、セット境界(リザルト演出~キャラ選択~新セット開始)を精密特定
set -e
cd "/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
SRC="data/frames/video_zenchi_c0BQoMJwwQU.mp4"
OUTDIR="data/verify/zenchi_boundary_confirm_2026-08-21/setgap_narrow"
mkdir -p "$OUTDIR"

extract() {
  t="$1"; name="$2"
  pre=$(awk -v t="$t" 'BEGIN{v=t-3; if(v<0)v=0; printf "%.3f", v}')
  off=$(awk -v t="$t" -v pre="$pre" 'BEGIN{printf "%.3f", t-pre}')
  ffmpeg -y -ss "$pre" -i "$SRC" -ss "$off" -frames:v 1 -q:v 2 "$OUTDIR/$name" >/dev/null 2>&1
}

for t in $(seq 3480 5 3700); do
  extract "$t" "sg_t${t}.png"
done
echo "done: $(ls "$OUTDIR" | wc -l) files"
