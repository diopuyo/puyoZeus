#!/usr/bin/env bash
# FINAL6 (連鎖表示ホールドなし) と FINAL7 (あり) の同一時刻フレームを抽出。
# 抽出時刻は診断で 2P が連鎖中に gravity_settle -> ojama_fall へ抜けていた
# 区間の中央 (56.2 / 57.6 / 59.0 / 60.35 / 61.7 / 63.0)。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
DEMO=data/verify/youtube_demo_2026-08-07
OUT="$DEMO/final6_vs_7"
mkdir -p "$OUT"
FF=$(./venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
for T in 56.2 57.6 59.0 60.35 61.7 63.0; do
  for V in FINAL9a_all_states; do
    "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" \
      -i "$DEMO/dio_vs_ts_${V}_viz.mp4" -frames:v 1 "$OUT/${V}_t${T}.png"
  done
done
ls "$OUT"
