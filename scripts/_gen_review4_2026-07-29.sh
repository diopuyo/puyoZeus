#!/bin/bash
# 4試合の有利不利オーバーレイ動画 + 校正ON比較1本 (2026-07-29)
#
# 目的: 打ち合い計測以外の直近タスク(extra4収集14並列 + Platt校正学習)が
#   完走したタイミングで、未レビューのマスター級動画4試合をフルレンダする。
#   CPU競合を避けるため、収集ジョブ完走まで本スクリプト自身が待機する
#   (CLAUDE.md プロセス管理ルール: setsid -f detach で起動し、本体はここで待つ)。
#
# 選定 (詳細はコーダ報告参照):
#   - c56/c60/c65/c75 (いずれも「第2回新おいうリーグ」マスター級、
#     data/_dl_expand.tsv でタイトル確認済み)。c56/c60/c65/c75 は
#     いずれも過去レビュー未使用 (c34=使用済み、c71=video_84として使用済みのため除外)。
#   - 境界は data/verify/winners_panel_diff_gated_2026-07-26/video_cNN.json の
#     confidence=="strict" な1ゲーム区間をそのまま採用 (勝敗パネル差分由来)。
#
# 校正 (enable_platt_calibration):
#   - 4試合は --no-platt-calibration で明示的に校正OFF(現状の既定動作)。
#     [重要な既知不整合] scripts/visualize_advantage_overlay.py の
#     generate() 関数既定は False (コミット0a0b014) だが、CLI (argparse) 側の
#     --no-platt-calibration は default=True のままで、フラグ無指定だと
#     enable_platt_calibration=True が generate() に渡り校正器ファイル必須に
#     なる (CalibrationFileMissingError)。本スクリプトはこの不整合を踏まない
#     ため、OFF/ON いずれの場合も明示的にフラグを与える。
#   - 校正器 data/indicators_v2/platt_calibration.json が存在する場合のみ、
#     c56 の1試合を校正ON(フラグ省略=CLI既定True)でも追加生成する。
#     存在しない場合はスキップしログに明記する(fail-silent にしない)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.

LOG="logs/gen_review4_2026-07-29.log"
OUT_DIR="data/verify/review4_2026-07-29"
CALIB="data/indicators_v2/platt_calibration.json"
mkdir -p "${OUT_DIR}" logs

log() { echo "[review4] $(date) $*" | tee -a "${LOG}"; }

log "起動 (PID=$$)。待機開始 (収集完走 + Platt校正学習完了待ち)。"

# --- 待機(a): extra4収集14並列 (_collect_1t) が消えるまで ---
while pgrep -f "_collect_1t" > /dev/null; do
  sleep 60
done
log "収集プロセス(_collect_1t)の消失を検知"

# --- 待機(b): Platt校正学習ラッパー (_wait_and_fit_platt) が消えるまで ---
#   このスクリプトは _collect_1t 消失後に fit_platt_calibration を実行し、
#   完了(または異常終了)すると自身も終了する。
while pgrep -f "_wait_and_fit_platt_2026-07-29" > /dev/null; do
  sleep 60
done
log "Platt校正待機スクリプトの消失を検知(学習完了 or 異常終了)"

sleep 60  # CPU安定待ち (直後の負荷変動を避ける、既存スクリプトと同じ余裕)
log "生成開始"

FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')

# 既定ON4種 (show-recognition/landing-observed-color/drift-guards/
# match-start-full-clear) + #51系3フラグ (recovery-counter-carryover/
# cnn-flicker-hsv-fallback/initial-confirm-vote) は CLI 既定が本体既定と
# 未同期のため明示指定必須 (video_84/c34 レビュー時と同じ理由)。
COMMON_FLAGS="--show-recognition --landing-observed-color --drift-guards --match-start-full-clear --recovery-counter-carryover --cnn-flicker-hsv-fallback --initial-confirm-vote --warmup-sec 30"

MIN_RAW_BYTES=$((1 * 1024 * 1024))       # 1MiB未満は異常 (過去実測は62-170MiB)
MIN_H264_BYTES=$((500 * 1024))           # 500KiB未満は異常

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
  local name="$1" video="$2" start="$3" end="$4" extra="$5"
  local raw="${OUT_DIR}/advantage_${name}_full_score0to0.mp4"
  local h264="${OUT_DIR}/advantage_${name}_full_score0to0_h264.mp4"
  log "[${name}] レンダ開始 video=${video} start=${start} end=${end} extra=[${extra}]"
  if [ ! -f "data/frames/${video}.mp4" ]; then
    log "ERROR: 入力動画が見つからない: data/frames/${video}.mp4"
    return 1
  fi
  nice -n 10 ./venv/bin/python -m scripts._zap_1t \
    --video "data/frames/${video}.mp4" --out "${raw}" \
    --start-sec "${start}" --end-sec "${end}" \
    ${COMMON_FLAGS} ${extra} >> "${LOG}" 2>&1
  if ! check_output "${raw}" "${MIN_RAW_BYTES}"; then
    log "[${name}] 異常終了 (raw未生成/過小、fail-silentにせず停止対象)"
    return 1
  fi
  "$FF" -y -loglevel error -i "${raw}" -c:v libx264 -preset medium -crf 20 \
    -pix_fmt yuv420p -movflags +faststart "${h264}" >> "${LOG}" 2>&1
  if ! check_output "${h264}" "${MIN_H264_BYTES}"; then
    log "[${name}] h264変換異常"
    return 1
  fi
  log "[${name}] 完了"
  return 0
}
export -f render_one check_output log
export LOG OUT_DIR COMMON_FLAGS FF MIN_RAW_BYTES MIN_H264_BYTES

# --- 4試合定義 (未レビュー・マスター級・4動画にまたがる、winners_panel由来境界) ---
JOBS=(
  "c56_g3|video_c56|288.0|362.0|--no-platt-calibration"
  "c60_g2|video_c60|314.0|400.0|--no-platt-calibration"
  "c65_g3|video_c65|306.0|440.0|--no-platt-calibration"
  "c75_g0|video_c75|200.0|268.0|--no-platt-calibration"
)

# --- 校正ON比較1本 (c56試合を再利用、校正器が存在する場合のみ追加) ---
# 【重要】校正ONは --platt-calibration を**明示**すること。
# 2026-07-29 の事故: 本スクリプトは当初「フラグ省略 = CLI既定 True = 校正ON」を前提に
# 空文字を渡していたが、その後 CLI 既定が False に修正され主フラグが --platt-calibration に
# 改名された (コミット b04eaec) ため、空文字は「校正OFF」を意味するようになった。
# 結果、校正ON版が校正なしで生成され OFF版と md5 完全一致の重複動画になった。
# 既定値に依存せず常に明示的に渡す。
if [ -f "${CALIB}" ]; then
  log "校正器 ${CALIB} を検出 -> c56_g3 の校正ON版も追加生成する"
  JOBS+=("c56_g3_calibON|video_c56|288.0|362.0|--platt-calibration")
else
  log "校正器 ${CALIB} が未生成 -> 校正ON版の生成をスキップする"
fi

FAIL=0
BATCH_SIZE=3
i=0
n=${#JOBS[@]}
while [ "$i" -lt "$n" ]; do
  batch_pids=()
  batch_end=$((i + BATCH_SIZE))
  j=$i
  while [ "$j" -lt "$batch_end" ] && [ "$j" -lt "$n" ]; do
    IFS='|' read -r name video start end extra <<< "${JOBS[$j]}"
    render_one "$name" "$video" "$start" "$end" "$extra" &
    batch_pids+=("$!")
    j=$((j + 1))
  done
  for pid in "${batch_pids[@]}"; do
    wait "$pid" || FAIL=1
  done
  i=$batch_end
done

if [ "$FAIL" -ne 0 ]; then
  log "ERROR DONE: 一部ジョブが失敗 (詳細はログ内 ERROR 行を参照)"
  exit 1
fi
log "ALL DONE"
