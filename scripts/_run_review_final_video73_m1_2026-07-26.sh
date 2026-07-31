#!/bin/bash
# 最終レビュー動画 (2026-07-26): 未見動画 video_73 (スラさん vs タイタン、マスター4ブロック)
# の match1 (matches.tsv start=135.0 end=317.0) をフル試合レンダ。
#
# --show-recognition のみでは新既定(2026-07-25承認4修正、コミットd6fffe3で
# RecognitionPipeline.load_default の既定 True 化)を再現できない。
# scripts/visualize_advantage_overlay.py の CLI 引数 (--landing-observed-color /
# --drift-guards / --match-start-full-clear) は default=False のまま据え置かれて
# おり、generate() がこれを明示的に load_default へ渡すため、d6fffe3 の
# 新既定 (True) を上書きしてしまう (scripts/visualize_advantage_overlay.py:843-849)。
# よって v6 (2026-07-25 承認版, scripts/_run_full_game1_c34_v6_2026-07-25.sh) と
# 同一構成で、4フラグ全てを明示指定する。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1
OUT_DIR="data/verify/review_video_final_2026-07-26"
mkdir -p "${OUT_DIR}"
RAW="${OUT_DIR}/advantage_recog_video73_match1_full_score0to0.mp4"
H264="${OUT_DIR}/advantage_recog_video73_match1_full_score0to0_h264.mp4"

echo "[start] $(date)"
PYTHONPATH=. nice -n 10 ./venv/bin/python -m scripts._zap_1t \
  --video data/frames/video_73.mp4 \
  --out "${RAW}" \
  --start-sec 135.0 --end-sec 317.0 --warmup-sec 30 \
  --show-recognition --landing-observed-color --drift-guards --match-start-full-clear
echo "[render done] $(date)"

FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
"$FF" -y -i "${RAW}" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "${H264}"
echo "[h264 done] $(date)"
ls -la "${RAW}" "${H264}"
echo "[all done] $(date)"
