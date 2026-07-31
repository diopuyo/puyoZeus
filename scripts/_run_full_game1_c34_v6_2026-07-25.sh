#!/bin/bash
# レビュー#49 再生成 v6 (2026-07-25): v4 の幽霊B(前試合残骸5field)修正版。
#
# v4 (コミット a06eeae) で「開始直後の幽霊B(前試合の残骸が5フィールド分
# 残る)」が確認されたため、RecognitionPipeline.load_default の
# enable_match_start_full_clear (前試合盤面残骸リーク修正、2026-07-23 実装済み
# だが未配線) を有効化して再検証する。
# 構成は v4 と同一 (465.6-511.8s、warmup 30s、着地色補正ON + Driftガード2種ON)
# に --match-start-full-clear のみ追加。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1
OUT_DIR="data/verify/review_video_new_2026-07-25"
mkdir -p "${OUT_DIR}"
RAW="${OUT_DIR}/advantage_recog_c34_game1_full_score0to0_v6.mp4"
H264="${OUT_DIR}/advantage_recog_c34_game1_full_score0to0_v6_h264.mp4"

echo "[start] $(date)"
PYTHONPATH=. nice -n 10 ./venv/bin/python -m scripts._zap_1t \
  --video data/frames/video_c34.mp4 \
  --out "${RAW}" \
  --start-sec 465.6 --end-sec 511.8 --warmup-sec 30 \
  --show-recognition --landing-observed-color --drift-guards --match-start-full-clear
echo "[render done] $(date)"

FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
"$FF" -y -i "${RAW}" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "${H264}"
echo "[h264 done] $(date)"
ls -la "${RAW}" "${H264}"
echo "[all done] $(date)"
