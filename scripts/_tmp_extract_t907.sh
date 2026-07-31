#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
IN="data/indicators_v2/overlay/advantage_recog_c62_estimated_board_chain_window_h264.mp4"
mkdir -p /tmp/frames_t907
# overlay動画は write_frame=900sから開始 → 動画内7秒=元動画t=907s
for off in 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0; do
  "$FF" -y -ss "$off" -i "$IN" -frames:v 1 "/tmp/frames_t907/overlay_off${off}.png" 2>/dev/null
done
# 生動画側 (認識overlayなしの元映像、ネイティブ1920x1080) も同時刻を抜く
RAWVIDEO="data/frames/video_c62.mp4"
for t in 905.5 906.0 906.5 907.0 907.5 908.0 908.5 909.0; do
  "$FF" -y -ss "$t" -i "$RAWVIDEO" -frames:v 1 "/tmp/frames_t907/raw_t${t}.png" 2>/dev/null
done
ls -la /tmp/frames_t907/
