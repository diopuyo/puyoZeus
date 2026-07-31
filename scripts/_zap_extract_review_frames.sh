#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
SRC="data/indicators_v2/overlay/zap/recognition_zap_reel_v2_2026-07-23.mp4"
OUT="data/indicators_v2/overlay/zap/frames"
mkdir -p "$OUT"

# 番号 名前 start dur (final動画内オフセット・秒)
SEGS=(
  "01 video_c5 0 25" "02 video_c8 25 15" "03 video_c12 40 25" "04 video_c15 65 15"
  "05 video_c17 80 15" "06 video_c20 95 25" "07 video_c23 120 15" "08 video_c28 135 25"
  "09 video_c31 160 15" "10 video_c40 175 25" "11 video_c45 200 15" "12 video_c50 215 15"
  "13 video_c58 230 15" "14 video_c65 245 25" "15 video_c70 270 15" "16 video_c78 285 15"
  "17 video_c95 300 15" "18 video_31 315 25" "19 video_33 340 15" "20 video_36 355 15"
  "21 video_37 370 25" "22 video_c82 395 25" "23 video_c83 420 25" "24 video_c84 445 25"
  "25 video_c85 470 15" "26 video_c86 485 15" "27 video_c89 500 15" "28 video_c92 515 15"
)
for seg in "${SEGS[@]}"; do
  read -r num name start dur <<< "$seg"
  t1=$(echo "$start + $dur * 0.35" | bc)
  t2=$(echo "$start + $dur * 0.85" | bc)
  "$FF" -y -ss "$t1" -i "$SRC" -frames:v 1 -update 1 "$OUT/${num}_${name}_a.png" > /dev/null 2>&1
  "$FF" -y -ss "$t2" -i "$SRC" -frames:v 1 -update 1 "$OUT/${num}_${name}_b.png" > /dev/null 2>&1
done
echo "done"
ls "$OUT" | grep -c png
