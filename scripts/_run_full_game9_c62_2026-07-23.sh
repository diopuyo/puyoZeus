#!/bin/bash
# c62 game9 フル試合(score0→次score0)通しレンダリング + h264変換。
# 境界は scripts/_tmp_find_score0_c62.py の実測(ScoreOcr直読み)で確定:
#   game9開始 score0 = t=872.4s (game8終値8805/4360 -> 872.2-872.3 None(遷移) -> 872.4 で0/0)
#   game10開始 score0 = t=949.5s (game9終値68444/84997 -> 949.3-949.4 None(遷移) -> 949.5 で0/0)
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1
RAW="data/indicators_v2/overlay/advantage_recog_c62_game9_full_score0to0.mp4"
H264="data/indicators_v2/overlay/advantage_recog_c62_game9_full_score0to0_h264.mp4"

echo "[start] $(date)"
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_advantage_overlay \
  --video data/frames/video_c62.mp4 \
  --out "${RAW}" \
  --start-sec 872.4 --end-sec 949.5 --warmup-sec 30 \
  --show-recognition
echo "[render done] $(date)"

FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
"$FF" -y -i "${RAW}" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "${H264}"
echo "[h264 done] $(date)"
ls -la "${RAW}" "${H264}"
echo "[all done] $(date)"
