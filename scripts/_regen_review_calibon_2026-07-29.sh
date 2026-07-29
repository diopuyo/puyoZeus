#!/bin/bash
# 校正ON版だけを再生成する (2026-07-29 の重複事故のリカバリ)。
#
# 事故の経緯: _gen_review4_2026-07-29.sh は「フラグ省略 = CLI既定True = 校正ON」を前提に
# 空文字を渡していたが、その後 CLI 既定が False に修正され主フラグが --platt-calibration に
# 改名された (コミット b04eaec) ため、空文字が「校正OFF」を意味するようになった。
# 結果、校正ON版が校正なしで生成され OFF版と md5 完全一致 (1ca6d64d...) の重複になった。
#
# 本スクリプトは --platt-calibration を明示して c56_g3 の校正ON版のみ作り直す。
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
export PYTHONPATH=.

LOG=logs/regen_review_calibon_2026-07-29.log
OUT_DIR=data/verify/review4_2026-07-29
NAME=c56_g3_calibON
RAW="${OUT_DIR}/advantage_${NAME}_full_score0to0.mp4"
H264="${OUT_DIR}/advantage_${NAME}_full_score0to0_h264.mp4"
CALIB=data/indicators_v2/platt_calibration.json
MIN_RAW_BYTES=1048576
MIN_H264_BYTES=512000

log() { echo "[regen-calibon] $(date) $*" | tee -a "$LOG"; }

log "起動 (PID=$$)"

if [ ! -f "$CALIB" ]; then
  log "[ERROR] 校正器が無い: $CALIB"
  log "ERROR DONE"
  exit 1
fi

# 重複していた旧ファイルを退避 (削除せず残す。比較の証跡として)
for f in "$RAW" "$H264"; do
  if [ -f "$f" ]; then
    mv "$f" "${f}.dup_uncalibrated_bak"
    log "重複ファイルを退避: ${f}.dup_uncalibrated_bak"
  fi
done

log "レンダ開始 (--platt-calibration を明示)"
nice -n 10 ./venv/bin/python -m scripts._zap_1t \
  -m scripts.visualize_advantage_overlay \
  --video data/frames/video_c56.mp4 \
  --out "$RAW" \
  --start-sec 288.0 --end-sec 362.0 \
  --show-recognition --landing-observed-color --drift-guards \
  --match-start-full-clear \
  --recovery-counter-carryover --cnn-flicker-hsv-fallback --initial-confirm-vote \
  --platt-calibration \
  >> "$LOG" 2>&1
log "レンダ終了"

# fail-silent 防止: サイズ検証
if [ ! -f "$RAW" ]; then
  log "[ERROR] raw が生成されていない: $RAW"
  log "ERROR DONE"
  exit 1
fi
sz=$(stat -c%s "$RAW")
if [ "$sz" -lt "$MIN_RAW_BYTES" ]; then
  log "[ERROR] raw が小さすぎる: $sz bytes"
  log "ERROR DONE"
  exit 1
fi
log "OK: $RAW ($sz bytes)"

# h264 変換
if [ -f "$H264" ]; then rm -f "$H264"; fi
nice -n 10 ./venv/bin/python -c "
import imageio_ffmpeg, subprocess, sys
exe = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([exe, '-y', '-i', '$RAW', '-c:v', 'libx264', '-preset', 'medium',
                '-crf', '23', '-pix_fmt', 'yuv420p', '$H264'], check=True)
" >> "$LOG" 2>&1 || { log "[ERROR] h264変換に失敗"; log "ERROR DONE"; exit 1; }

sz2=$(stat -c%s "$H264")
if [ "$sz2" -lt "$MIN_H264_BYTES" ]; then
  log "[ERROR] h264 が小さすぎる: $sz2 bytes"
  log "ERROR DONE"
  exit 1
fi
log "OK: $H264 ($sz2 bytes)"

# 校正OFF版との差分確認 (同一なら校正が効いていない = 事故の再発)
OFF_H264="${OUT_DIR}/advantage_c56_g3_full_score0to0_h264.mp4"
if [ -f "$OFF_H264" ]; then
  m1=$(md5sum "$OFF_H264" | cut -d' ' -f1)
  m2=$(md5sum "$H264" | cut -d' ' -f1)
  log "md5 OFF=$m1 ON=$m2"
  if [ "$m1" = "$m2" ]; then
    log "[ERROR] 校正ON版がOFF版と同一。校正が効いていない (事故の再発)"
    log "ERROR DONE"
    exit 1
  fi
  log "差分あり = 校正が効いている"
fi

log "ALL DONE"
