#!/usr/bin/env bash
# 最終2本のスマホ視聴用 (720p / 音声なし / faststart)。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
D=data/verify/youtube_demo_2026-08-07/release
OUT="$D/mobile"
mkdir -p "$OUT"
rm -f "$OUT"/*.mp4
FF=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
enc() {
  "$FF" -nostdin -hide_banner -loglevel error -y -i "$1" -vf "scale=-2:720"     -c:v libx264 -preset veryfast -crf 28 -pix_fmt yuv420p -movflags +faststart -an "$2"
  echo "[mobile] $(basename $2) $(stat -c %s "$2") bytes"
}
enc "$D/final_1_full_overlay.mp4" "$OUT/final_1_full_overlay_720p.mp4"
enc "$D/final_2_stable_tsumo.mp4"  "$OUT/final_2_stable_tsumo_720p.mp4"
