#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
FFPROBE="${FF%ffmpeg-linux-x86_64-v7.0.2}ffprobe-linux-x86_64-v7.0.2"
ORDER=(01_video_c5 02_video_c8 03_video_c12 04_video_c15 05_video_c17 06_video_c20 07_video_c23 08_video_c28 09_video_c31 10_video_c40 11_video_c45 12_video_c50 13_video_c58 14_video_c65 15_video_c70 16_video_c78 17_video_c95 18_video_31 19_video_33 20_video_36 21_video_37 22_video_c82 23_video_c83 24_video_c84 25_video_c85 26_video_c86 27_video_c89 28_video_c92)
for name in "${ORDER[@]}"; do
  f="data/indicators_v2/overlay/zap/labeled/${name}_h264.mp4"
  dur=$("$FFPROBE" -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$f" 2>/dev/null)
  echo "$name $dur"
done
