#!/bin/bash
# レビュー#49 (2026-07-25): 未レビュー新規動画 video_c34 game1 のフル試合
# 認識レビュー動画 (score0→次score0通し) + h264変換。
# #41 (c62 game9) と同形式=scripts.visualize_advantage_overlay --show-recognition。
# enable_landing_observed_color=True を明示指定 (今回の採用候補修正を反映)。
#
# 境界は scripts/_tmp_find_score0_c34.py の実測 (ScoreOcr直読み) で確定:
#   game1開始 score0 = t=465.6s (game0終値27572/6113 -> 465.4s None(遷移) -> 465.6s で0/0)
#   game2開始 score0 = t=511.8s (game1終値24164/4720 -> 511.4s None(遷移) -> 511.6-511.8s で0/0)
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1
OUT_DIR="data/verify/review_video_new_2026-07-25"
mkdir -p "${OUT_DIR}"
RAW="${OUT_DIR}/advantage_recog_c34_game1_full_score0to0.mp4"
H264="${OUT_DIR}/advantage_recog_c34_game1_full_score0to0_h264.mp4"

echo "[start] $(date)"
# scripts/_zap_1t.py = cv2.setNumThreads(1) ラッパー (scripts.visualize_advantage_overlay.main を
# そのまま呼ぶだけ、src/は無改修)。熱対策のため1本のみでもスレッド固定して回す。
PYTHONPATH=. nice -n 10 ./venv/bin/python -m scripts._zap_1t \
  --video data/frames/video_c34.mp4 \
  --out "${RAW}" \
  --start-sec 465.6 --end-sec 511.8 --warmup-sec 30 \
  --show-recognition --landing-observed-color
echo "[render done] $(date)"

FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
"$FF" -y -i "${RAW}" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "${H264}"
echo "[h264 done] $(date)"
ls -la "${RAW}" "${H264}"
echo "[all done] $(date)"
