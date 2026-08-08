#!/usr/bin/env bash
# FINAL5 (修正なし) と FINAL6 (振動バグ B+C 修正) の同一時刻フレームを抽出して
# 目視比較用に並べる。抽出時刻は baseline 診断で視覚由来の誤 OJAMA_FALL が
# 出ていた t=56.3 / 57.7 と、対照として t=61.7。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
DEMO=data/verify/youtube_demo_2026-08-07
OUT="$DEMO/final5_vs_6"
mkdir -p "$OUT"
FF=$(./venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
for T in 56.3 57.7 61.7; do
  for V in FINAL5a_all_states FINAL6a_all_states; do
    "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" \
      -i "$DEMO/dio_vs_ts_${V}_viz.mp4" -frames:v 1 "$OUT/${V}_t${T}.png"
  done
done
ls -la "$OUT"
