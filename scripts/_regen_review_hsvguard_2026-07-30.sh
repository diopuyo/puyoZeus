#!/bin/bash
# 色→空 HSV照合ガード (enable_puyo_to_empty_hsv_guard, コミット 97445cc) の
# user承認レビュー動画一式を生成する (2026-07-30)。
#
# 選定動画: video_c10 (data/verify/winners_panel_diff_2026-07-26/video_c10.json)
#   game_abs_idx=9, start_sec=758.0, end_sec=816.0 (58秒、フル試合分)。
# 選定基準 (report参照):
#   - 未見の新動画 (c34/c56/c60/c65/c75/video_84 は使用済のため除外)
#   - scripts/select_labeled_win_videos.py の _ALLOWED_C_INDEX_RANGES で
#     ティア確認済 (4-33=チャレンジャー) の範囲から選定
#   - data/verify/winners_panel_diff_2026-07-26/video_c*.json のうち
#     confidence=="strict" かつ game_abs_idx>=3 (動画冒頭を避ける) かつ
#     試合長55-65秒 (先行レビュー動画と同程度) を満たす最初の動画 (video_c10) の
#     最初の該当試合 (game_abs_idx=9) を機械的に採用 (目視による恣意選択を排除)。
#
# OFF版・ON版は既存レビュー動画 (_regen_review_calibon_2026-07-29.sh) と同じ
# 主要フラグ一式 + ON版のみ --puyo-to-empty-hsv-guard を追加する。
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
export PYTHONPATH=.

LOG=logs/regen_review_hsvguard_2026-07-30.log
OUT_DIR=data/verify/review_hsvguard_2026-07-30
VIDEO=data/frames/video_c10.mp4
START_SEC=758.0
END_SEC=816.0
WARMUP_SEC=30
CALIB=data/indicators_v2/platt_calibration.json
MIN_RAW_BYTES=1048576
MIN_H264_BYTES=512000
EXPECTED_FILES=4   # raw(OFF/ON) + h264(OFF/ON)

OFF_NAME=c10_g9_hsvguardOFF
ON_NAME=c10_g9_hsvguardON
OFF_RAW="${OUT_DIR}/advantage_${OFF_NAME}_full.mp4"
ON_RAW="${OUT_DIR}/advantage_${ON_NAME}_full.mp4"
OFF_H264="${OUT_DIR}/advantage_${OFF_NAME}_full_h264.mp4"
ON_H264="${OUT_DIR}/advantage_${ON_NAME}_full_h264.mp4"

log() { echo "[regen-hsvguard] $(date) $*" | tee -a "$LOG"; }

log "起動 (PID=$$)"

if [ ! -f "$CALIB" ]; then
  log "[ERROR] 校正器が無い: $CALIB"
  log "ERROR DONE"
  exit 1
fi
if [ ! -f "$VIDEO" ]; then
  log "[ERROR] 動画が無い: $VIDEO"
  log "ERROR DONE"
  exit 1
fi

mkdir -p "$OUT_DIR"

_render() {
  local out_raw="$1"; shift
  local extra_flag="$1"; shift
  log "レンダ開始: $out_raw (extra=${extra_flag:-なし})"
  nice -n 19 ./venv/bin/python -m scripts._zap_1t \
    --video "$VIDEO" \
    --out "$out_raw" \
    --start-sec "$START_SEC" --end-sec "$END_SEC" \
    --warmup-sec "$WARMUP_SEC" \
    --show-recognition --landing-observed-color --drift-guards \
    --match-start-full-clear \
    --recovery-counter-carryover --cnn-flicker-hsv-fallback --initial-confirm-vote \
    --platt-calibration \
    $extra_flag \
    >> "$LOG" 2>&1
  if [ ! -f "$out_raw" ]; then
    log "[ERROR] raw が生成されていない: $out_raw"
    log "ERROR DONE"
    exit 1
  fi
  sz=$(stat -c%s "$out_raw")
  if [ "$sz" -lt "$MIN_RAW_BYTES" ]; then
    log "[ERROR] raw が小さすぎる: $out_raw ($sz bytes)"
    log "ERROR DONE"
    exit 1
  fi
  log "OK: $out_raw ($sz bytes)"
}

_to_h264() {
  local raw="$1"; local h264="$2"
  if [ -f "$h264" ]; then rm -f "$h264"; fi
  nice -n 19 ./venv/bin/python -c "
import imageio_ffmpeg, subprocess
exe = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([exe, '-y', '-i', '$raw', '-c:v', 'libx264', '-preset', 'medium',
                '-crf', '23', '-pix_fmt', 'yuv420p', '$h264'], check=True)
" >> "$LOG" 2>&1 || { log "[ERROR] h264変換に失敗: $raw"; log "ERROR DONE"; exit 1; }
  sz2=$(stat -c%s "$h264")
  if [ "$sz2" -lt "$MIN_H264_BYTES" ]; then
    log "[ERROR] h264 が小さすぎる: $h264 ($sz2 bytes)"
    log "ERROR DONE"
    exit 1
  fi
  log "OK: $h264 ($sz2 bytes)"
}

# OFF版 (既定挙動、新フラグなし)
_render "$OFF_RAW" ""
_to_h264 "$OFF_RAW" "$OFF_H264"

# ON版 (新フラグ有効)
_render "$ON_RAW" "--puyo-to-empty-hsv-guard"
_to_h264 "$ON_RAW" "$ON_H264"

# fail-silent 防止: md5 差分確認 (同一なら新フラグが効いていない = 事故)
m1=$(md5sum "$OFF_H264" | cut -d' ' -f1)
m2=$(md5sum "$ON_H264" | cut -d' ' -f1)
log "md5 OFF=$m1 ON=$m2"
if [ "$m1" = "$m2" ]; then
  log "[ERROR] ON版がOFF版と同一。フラグが効いていない (事故)"
  log "ERROR DONE"
  exit 1
fi
log "差分あり = フラグが効いている"

# fail-silent 防止: 期待ファイル件数確認 (raw2 + h264 2 = 4)
actual=$(ls -1 "$OUT_DIR"/advantage_c10_g9_hsvguard*.mp4 2>/dev/null | wc -l)
log "件数確認: actual=$actual expected=$EXPECTED_FILES"
if [ "$actual" -ne "$EXPECTED_FILES" ]; then
  log "[ERROR] 件数不一致 (actual=$actual expected=$EXPECTED_FILES)"
  log "ERROR DONE"
  exit 1
fi

log "ALL DONE"
