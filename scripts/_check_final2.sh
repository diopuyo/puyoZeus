#!/usr/bin/env bash
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
D=data/verify/youtube_demo_2026-08-07/release
OUT="$D/check"
mkdir -p "$OUT"
FF=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
for T in 26.2 57.6; do
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" -i "$D/final_1_with_forecast.mp4" -frames:v 1 "$OUT/f1_t${T}.png"
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" -i "$D/final_2_stable_tsumo.mp4" -frames:v 1 "$OUT/f2_t${T}.png"
done
ls "$OUT"
