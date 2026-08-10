#!/usr/bin/env bash
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
DEMO=data/verify/youtube_demo_2026-08-07
OUT="$DEMO/cap_check"
mkdir -p "$OUT"
FF=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
for T in 29.0 45.0 54.5 62.0 66.0; do
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" -i "$DEMO/_ab_capability_pressure.mp4" -frames:v 1 "$OUT/cap_t${T}.png"
done
ls "$OUT"
