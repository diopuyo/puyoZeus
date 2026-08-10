#!/usr/bin/env bash
# user 指摘「26秒の 1P 2連鎖なのに 1連鎖判定」の実画面確認。
# 連鎖は数フレームで進むため、 前後 1 秒を 0.2 秒刻みで抜く。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
DEMO=data/verify/youtube_demo_2026-08-07
OUT="$DEMO/t26_check"
mkdir -p "$OUT"
FF=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
for T in 25.4 25.8 26.0 26.2 26.4 26.6 27.0; do
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" \
    -i "$DEMO/dio_vs_ts_FINAL10a_all_states_viz.mp4" -frames:v 1 "$OUT/FINAL10a_t${T}.png"
done
ls "$OUT"
