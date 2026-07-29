#!/bin/bash
# B/C 版生成 (2026-07-29): 「早くなった」の原因が校正かwarmupかを1条件ずつ
# 切り分けるための短尺版。区間は user 指定の abs 335〜362秒 (該当区間指摘の
# 前後を含む27秒)。既存の A (calibOFF/warmupあり) ・ D (calibON/warmupなし)
# は削除・上書きしない (別ファイル名で生成)。
#   B: 校正ON + warmupあり (--warmup-sec 30) -> Aとの差分=校正だけの効果
#   C: 校正OFF + warmupなし                 -> Aとの差分=warmupだけの効果
# nice -n 19、逐次 (B完了後にC、並列にしない)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.

LOG=logs/gen_review_bc_2026-07-29.log
OUT_DIR=data/verify/review4_2026-07-29
CALIB=data/indicators_v2/platt_calibration.json
mkdir -p "${OUT_DIR}" logs

log() { echo "[review-bc] $(date) $*" | tee -a "${LOG}"; }

log "起動 (PID=$$)"

if [ ! -f "${CALIB}" ]; then
  log "ERROR: 校正器が無い: ${CALIB} -> B版(校正ON)は生成不可、中止"
  log "ERROR DONE"
  exit 1
fi

FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')

COMMON_FLAGS="--show-recognition --landing-observed-color --drift-guards --match-start-full-clear --recovery-counter-carryover --cnn-flicker-hsv-fallback --initial-confirm-vote"

MIN_RAW_BYTES=$((1 * 1024 * 1024))
MIN_H264_BYTES=$((300 * 1024))

check_output() {
  local f="$1" min="$2"
  if [ ! -f "$f" ]; then
    log "ERROR: 出力が生成されていない: $f"
    return 1
  fi
  local sz
  sz=$(stat -c %s "$f")
  if [ "$sz" -lt "$min" ]; then
    log "ERROR: 出力サイズが異常に小さい (${sz} bytes < ${min}): $f"
    return 1
  fi
  log "OK: $f (${sz} bytes)"
  return 0
}

render_one() {
  local name="$1" start="$2" end="$3" extra="$4"
  local raw="${OUT_DIR}/advantage_${name}_score0to0.mp4"
  local h264="${OUT_DIR}/advantage_${name}_score0to0_h264.mp4"
  log "[${name}] レンダ開始 start=${start} end=${end} extra=[${extra}]"
  nice -n 19 ./venv/bin/python -u -m scripts._zap_1t \
    --video "data/frames/video_c56.mp4" --out "${raw}" \
    --start-sec "${start}" --end-sec "${end}" \
    ${COMMON_FLAGS} ${extra} >> "${LOG}" 2>&1
  if ! check_output "${raw}" "${MIN_RAW_BYTES}"; then
    log "[${name}] 異常終了 (raw未生成/過小)"
    return 1
  fi
  nice -n 19 "$FF" -y -loglevel error -i "${raw}" -c:v libx264 -preset medium -crf 20 \
    -pix_fmt yuv420p -movflags +faststart "${h264}" >> "${LOG}" 2>&1
  if ! check_output "${h264}" "${MIN_H264_BYTES}"; then
    log "[${name}] h264変換異常"
    return 1
  fi
  log "[${name}] 完了"
  return 0
}

# B: 校正ON + warmupあり (--warmup-sec 30) -> Aとの差 = 校正だけ
render_one "c56_g3_B_calibON_warmup30" 335.0 362.0 "--warmup-sec 30 --platt-calibration"

# C: 校正OFF + warmupなし -> Aとの差 = warmupだけ
render_one "c56_g3_C_calibOFF_warmup0" 335.0 362.0 "--no-platt-calibration"

log "ALL DONE"
