#!/bin/bash
# セット境界の精密特定: (A)WIN30到達直前の1秒刻み (B)セット2ゲーム開始直前の1秒刻み
set -e
cd "/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
SRC="data/frames/video_zenchi_c0BQoMJwwQU.mp4"
OUTDIR="data/verify/zenchi_boundary_confirm_2026-08-21/setboundary_precise"
mkdir -p "$OUTDIR"

extract() {
  t="$1"; name="$2"
  pre=$(awk -v t="$t" 'BEGIN{v=t-3; if(v<0)v=0; printf "%.3f", v}')
  off=$(awk -v t="$t" -v pre="$pre" 'BEGIN{printf "%.3f", t-pre}')
  ffmpeg -y -ss "$pre" -i "$SRC" -ss "$off" -frames:v 1 -q:v 2 "$OUTDIR/$name" >/dev/null 2>&1
}

# (A) WIN30到達 (t3485で28-29、t3490でWINNER画面、t3495で28-30表示)
for t in $(seq 3485 1 3496); do
  extract "$t" "A_t${t}.png"
done

# (B) セット2ゲーム開始 (t3650でNOW LOADING、t3655で既に対戦中score16/14)
for t in $(seq 3648 1 3660); do
  extract "$t" "B_t${t}.png"
done

echo "done: $(ls "$OUTDIR" | wc -l) files"
