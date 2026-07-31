#!/bin/bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
IN="data/verify/review_video_51flags_2026-07-27/advantage_recog_video84_match1_full_score0to0_h264_small.mp4"
OUTDIR="data/verify/review_video_51flags_2026-07-27/frames"
mkdir -p "${OUTDIR}"
for t in 76 79 82; do
  "${FF}" -y -loglevel error -ss "${t}" -i "${IN}" -frames:v 1 "${OUTDIR}/frame_t${t}s.png"
done
ls -la "${OUTDIR}"
