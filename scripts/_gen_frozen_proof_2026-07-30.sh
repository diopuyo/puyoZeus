#!/bin/bash
# 列消失バグが「描画経路でも見えるか」を決定的に検証するレンダ (2026-07-30)
#
# 経緯:
#   userが c60_g2 (314-400秒) と c75_g0 (200-268秒) を目視し「列がまるごと空白になる現象は
#   起きていない」と回答した。npz側でも同区間のE型候補は0件で一致した (分岐1)。
#   **しかしこの一致には検出力が無い**: E型候補の発生頻度は c60 で120秒に1件、c75 で459秒に1件で、
#   userが見た窓での期待値は合計0.87件。0件が観測される確率は42% (Poisson) なので、
#   「描画経路が健全」の証拠にはならない。
#
# そこで **npzがE型を検出している時刻を含む試合をレンダ**して直接確かめる。
#   - 見えれば: 描画経路にも同じ不具合がある = userが偶然見逃していただけ
#   - 見えなければ: **npz生成経路と描画経路で挙動が違う**という別の重大問題が確定
#
# 対象 (凍結が長く目視しやすい順に2件、試合境界はscoreリセットから特定):
#   c60: 2P col3 が t=1467.4 で崩壊し 20.8秒 凍結 → 試合 1428.2〜1498.6秒 (70秒)
#   c56: 1P col0 が t=2675.2 で崩壊し 38.6秒 凍結 → 試合 2653.2〜2725.0秒 (72秒)
# いずれも目標イベントが試合の中盤に位置する通し1本 (短窓禁止の規約を満たす)。
#
# フラグは userがレビュー済みの動画と同一にする (比較可能性のため)。
# warmup=30 は無害でむしろ初期収束に有利と実測済み、校正ONは盤面表示に影響しないと実フレームで確認済み。

set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

OUT_DIR=data/verify/frozen_proof_2026-07-30
LOG=logs/gen_frozen_proof_2026-07-30.log
CALIB=data/indicators_v2/platt_calibration.json
MIN_RAW_BYTES=5000000
MIN_H264_BYTES=500000

mkdir -p "$OUT_DIR"

log() { echo "[frozen-proof] $(date) $*" >> "$LOG"; }

log "=== 開始 ==="
if [ ! -f "$CALIB" ]; then
  log "[ERROR] 校正器が無い: $CALIB"
  log "ERROR DONE"
  exit 1
fi

COMMON="--show-recognition --landing-observed-color --drift-guards \
--match-start-full-clear --recovery-counter-carryover \
--cnn-flicker-hsv-fallback --initial-confirm-vote \
--warmup-sec 30 --platt-calibration"

render_one() {
  local name="$1" video="$2" start="$3" end="$4" note="$5"
  local raw="${OUT_DIR}/${name}.mp4"
  local h264="${OUT_DIR}/${name}_h264.mp4"

  log "[$name] レンダ開始 ${start}〜${end}秒 / 注目=${note}"
  # shellcheck disable=SC2086
  nice -n 10 ./venv/bin/python -m scripts._zap_1t \
    --video "data/frames/${video}.mp4" \
    --out "$raw" \
    --start-sec "$start" --end-sec "$end" \
    $COMMON >> "$LOG" 2>&1
  if [ ! -f "$raw" ]; then
    log "[$name][ERROR] raw が生成されていない"
    return 1
  fi
  local sz
  sz=$(stat -c%s "$raw")
  if [ "$sz" -lt "$MIN_RAW_BYTES" ]; then
    log "[$name][ERROR] raw が小さすぎる: $sz bytes"
    return 1
  fi
  log "[$name] OK raw ($sz bytes)"

  [ -f "$h264" ] && rm -f "$h264"
  nice -n 10 ./venv/bin/python -c "
import imageio_ffmpeg, subprocess
exe = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([exe, '-y', '-i', '$raw', '-c:v', 'libx264', '-preset', 'medium',
                '-crf', '23', '-pix_fmt', 'yuv420p', '$h264'], check=True)
" >> "$LOG" 2>&1 || { log "[$name][ERROR] h264変換に失敗"; return 1; }

  local sz2
  sz2=$(stat -c%s "$h264")
  if [ "$sz2" -lt "$MIN_H264_BYTES" ]; then
    log "[$name][ERROR] h264 が小さすぎる: $sz2 bytes"
    return 1
  fi
  log "[$name] OK h264 ($sz2 bytes)"
  return 0
}

# c60: 2P の col3 (左から4列目) が t=1467.4 から 20.8秒 空白になるはず
render_one "frozen_c60_t1467_2Pcol3" "video_c60" 1428.2 1498.6 \
  "2P col3 が 1467.4秒から20.8秒空白" || log "[c60] 失敗"

# c56: 1P の col0 (左端) が t=2675.2 から 38.6秒 空白になるはず
render_one "frozen_c56_t2675_1Pcol0" "video_c56" 2653.2 2725.0 \
  "1P col0 が 2675.2秒から38.6秒空白" || log "[c56] 失敗"

log "=== ALL DONE ==="
