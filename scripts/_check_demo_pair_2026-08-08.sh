#!/usr/bin/env bash
# YouTube デモ 2 本 (映像A=有利不利のみ / 映像B=認識オーバーレイ付き) の検収用に
# 同一時刻のフレームを抜き出す。 t=57.6 は 1P が 9 連鎖を撃っている最中。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
DEMO=data/verify/youtube_demo_2026-08-07
OUT="$DEMO/demo_pair_check"
mkdir -p "$OUT"
FF=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
for T in 20.0 45.0 57.6 70.0; do
  for V in A_advantage_only B_advantage_with_recognition; do
    "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" \
      -i "$DEMO/demo_${V}_2026-08-08.mp4" -frames:v 1 "$OUT/${V}_t${T}.png"
  done
done
ls -la "$OUT"
