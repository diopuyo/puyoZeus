#!/bin/bash
# レビュー#49 再生成 v4 (2026-07-25): video_c34 game1 の真の開始点版。
#
# v3 (start 472.0s = winners_probe strict 境界) は user 指摘「5手目からに見える」で
# 実フレーム検証した結果、境界診断の誤りが確定:
#   - 465.6s: 両盤面空 + スコア 00000000 = 真の試合開始待機
#   - 469.5s: 既に1手目設置済み (スコア25 = ドロップボーナス)
#   - 471.5s: 3-4手設置済み (スコア50/49)
#   → 472.0s は実開始より約5秒遅く、最初の4-5手が欠落していた。
# 「465.6-471.5s はロード演出・装飾アイコン」という v2/v3 の診断は誤り(実プレイ)。
# v1 の「開始直後の青2個」は幽霊ではなく実設置ぷよの誤認の可能性が高い。
# → v4 は 465.6s (スコアリセット・空盤面) 起点で冒頭を隠さず映す。
# 構成は v3 と同一: 着地色補正ON + Driftガード2種ON。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1
OUT_DIR="data/verify/review_video_new_2026-07-25"
mkdir -p "${OUT_DIR}"
RAW="${OUT_DIR}/advantage_recog_c34_game1_full_score0to0_v4.mp4"
H264="${OUT_DIR}/advantage_recog_c34_game1_full_score0to0_v4_h264.mp4"

echo "[start] $(date)"
PYTHONPATH=. nice -n 10 ./venv/bin/python -m scripts._zap_1t \
  --video data/frames/video_c34.mp4 \
  --out "${RAW}" \
  --start-sec 465.6 --end-sec 511.8 --warmup-sec 30 \
  --show-recognition --landing-observed-color --drift-guards
echo "[render done] $(date)"

FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
"$FF" -y -i "${RAW}" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "${H264}"
echo "[h264 done] $(date)"
ls -la "${RAW}" "${H264}"
echo "[all done] $(date)"
