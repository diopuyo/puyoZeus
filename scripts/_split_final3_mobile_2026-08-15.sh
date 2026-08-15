#!/bin/bash
# 最終デモ (final3, 指摘13込み) を試合ごとに分割し720pスマホ用mp4を作る。
# 分割点は _split_final2_mobile_2026-08-14.sh と同一 (0-56/56-116/116-148秒)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
SRC=data/verify/demo_fixed_2026-08-13/demo_final3_3match.mp4
OUT=data/verify/demo_fixed_2026-08-13
FF=$(./venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")

enc() {
  local ss="$1" to="$2" out="$3"
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$ss" -to "$to" -i "$SRC" \
    -vf "scale=-2:720" -c:v libx264 -preset veryfast -crf 28 -pix_fmt yuv420p \
    -movflags +faststart -an "$out"
  echo "[mobile] $(basename "$out") $(stat -c %s "$out") bytes"
}

enc 0 56 "$OUT/final3_m1.mp4"
enc 56 116 "$OUT/final3_m2.mp4"
enc 116 148 "$OUT/final3_m3.mp4"
echo SPLIT_FINAL3_DONE
