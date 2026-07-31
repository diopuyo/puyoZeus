#!/bin/bash
# #51+初回確定ガード user目視レビュー用動画 (2026-07-27)
# 未見のマスター級動画 video_84 (light vs ぷにちゃん、マスター3ブロック) の
# match1 (matches.tsv start=305.0 end=437.0) をフル試合レンダ。
#
# 既定ON4種 (show-recognition/landing-observed-color/drift-guards/
# match-start-full-clear) はCLI既定が本体既定値と未同期のため明示指定必須
# (2026-07-26 video_73レビュー時と同じ理由、_run_review_final_video73_m1_2026-07-26.sh
# 参照)。加えて #51 系3フラグ (recovery-counter-carryover/
# cnn-flicker-hsv-fallback/initial-confirm-vote) をON指定する
# (scripts/visualize_advantage_overlay.py に新規配線、コミット済)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1
OUT_DIR="data/verify/review_video_51flags_2026-07-27"
mkdir -p "${OUT_DIR}"
RAW="${OUT_DIR}/advantage_recog_video84_match1_full_score0to0.mp4"
H264="${OUT_DIR}/advantage_recog_video84_match1_full_score0to0_h264.mp4"
SMALL="${OUT_DIR}/advantage_recog_video84_match1_full_score0to0_h264_small.mp4"

echo "[start] $(date)"
PYTHONPATH=. nice -n 10 ./venv/bin/python -m scripts._zap_1t \
  --video data/frames/video_84.mp4 \
  --out "${RAW}" \
  --start-sec 305.0 --end-sec 437.0 --warmup-sec 30 \
  --show-recognition --landing-observed-color --drift-guards --match-start-full-clear \
  --recovery-counter-carryover --cnn-flicker-hsv-fallback --initial-confirm-vote
echo "[render done] $(date)"

FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
"$FF" -y -i "${RAW}" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "${H264}"
echo "[h264 crf20 done] $(date)"
ls -la "${RAW}" "${H264}"

# 30MiB超なら crf 28 で再エンコード
SIZE_BYTES=$(stat -c %s "${H264}")
LIMIT_BYTES=$((30 * 1024 * 1024))
if [ "${SIZE_BYTES}" -gt "${LIMIT_BYTES}" ]; then
  echo "[resize] h264 が30MiB超 (${SIZE_BYTES} bytes) -> crf28で再エンコード"
  "$FF" -y -loglevel error -i "${H264}" \
    -c:v libx264 -preset medium -crf 28 -pix_fmt yuv420p -movflags +faststart \
    "${SMALL}"
  ls -la "${SMALL}"
fi
echo "[all done] $(date)"
