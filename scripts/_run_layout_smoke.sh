#!/bin/bash
# 上部パネルレイアウト改修のスモーク検証(1クリップ)。
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. nice -n 10 ./venv/bin/python -m scripts.visualize_advantage_overlay \
    --video /home/ryouj/frames/video_30.mp4 \
    --out /tmp/layout_smoke.mp4 \
    --start-sec 2860 --end-sec 2872 --warmup-sec 16 \
    --exclude-video video_30
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
"$FF" -y -ss 6 -i /tmp/layout_smoke.mp4 -vframes 1 /tmp/layout_frame.png
"$FF" -i /tmp/layout_frame.png 2>&1 | grep Stream || true
ls -la /tmp/layout_frame.png
