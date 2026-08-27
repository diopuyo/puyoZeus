#!/bin/bash
# デモ2 (demo2v2, video_74 3試合) を試合ごとに分割し720pスマホ用mp4を作る。
# 分割点は既存の demo2_m1/m2/m3 (旧v4) で確立済みの試合境界と同一
# (0-56s=1試合目, 56-112s=2試合目, 112-177s(末尾)=3試合目、合計177s)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
SRC=data/verify/demo_fixed_2026-08-13/demo2_video74_3match.mp4
OUT=data/verify/demo_fixed_2026-08-13
FF=$(./venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")

enc() {
  local ss="$1" to="$2" out="$3"
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$ss" -to "$to" -i "$SRC" \
    -vf "scale=-2:720" -c:v libx264 -preset veryfast -crf 28 -pix_fmt yuv420p \
    -movflags +faststart -an "$out"
  echo "[mobile] $(basename "$out") $(stat -c %s "$out") bytes"
}

enc 0 56 "$OUT/demo2v2_m1.mp4"
enc 56 112 "$OUT/demo2v2_m2.mp4"
enc 112 177 "$OUT/demo2v2_m3.mp4"
