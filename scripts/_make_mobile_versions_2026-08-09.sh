#!/usr/bin/env bash
# スマホ視聴用に軽量化する (2026-08-09 user要望)。
#   1) 認識デモ        demo_final_C_recognition.mp4
#   2) 圧力なし指標デモ demo_no_pressure.mp4
# 720p / H.264 High / CRF 28 / faststart (先頭から再生できる) へ変換。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
DEMO=data/verify/youtube_demo_2026-08-07
OUT="$DEMO/mobile"
mkdir -p "$OUT"
FF=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
enc() {
  "$FF" -nostdin -hide_banner -loglevel error -y -i "$1"     -vf "scale=-2:720" -c:v libx264 -preset veryfast -crf 28     -pix_fmt yuv420p -movflags +faststart -an "$2"
  echo "[mobile] $(basename $2) $(stat -c %s "$2") bytes"
}
enc "$DEMO/demo_final_C_recognition.mp4"  "$OUT/mobile_recognition.mp4"
enc "$DEMO/demo_no_pressure.mp4"          "$OUT/mobile_advantage_no_pressure.mp4"
ls -la --time-style=+%H:%M "$OUT"
