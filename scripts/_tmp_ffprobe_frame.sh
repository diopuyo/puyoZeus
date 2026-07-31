#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
echo "FF=$FF"
IN="data/indicators_v2/overlay/advantage_v29_saturated_check.mp4"
# 中盤あたり(30秒地点)のフレームを1枚抜く
"$FF" -y -ss 30 -i "$IN" -vframes 1 /tmp/sat_frame.png 2>/dev/null
ls -la /tmp/sat_frame.png
