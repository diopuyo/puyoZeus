#!/bin/bash
# 間引きが盤面を壊す割合を定量する (2026-07-30)
#
# 経緯: c56 2P col1 で間引きあり=0個 / 全フレーム=8-9個 と確定したが n=1 で規模が不明。
# c60 では差が出なかったので、壊れる頻度を偏りなく測る必要がある。
#
# 設計: 候補が見つかった箇所を狙わず、各動画の**固定オフセット**の窓を取る(偏りを避ける)。
# 間引き以外の条件は完全に同一(開始時刻・長さ・フラグ)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
OUT=data/indicators_v2/sampling_rate_2026-07-30
LOG=logs/ab_sampling_rate_2026-07-30.log
mkdir -p "$OUT"
log() { echo "[ab-rate] $(date) $*" >> "$LOG"; }

log "=== 開始 ==="
# 各動画とも試合中盤に当たりやすい固定オフセットを選ぶ (候補位置を狙わない)
run_video() {
  local vid="$1" start="$2"
  for spec in "sampled|0.2" "allframes|0"; do
    local name="${spec%%|*}" si="${spec##*|}"
    nice -n 15 ./venv/bin/python -u -m scripts._collect_lean_1t \
      --video "data/frames/video_${vid}.mp4" \
      --out-npz "${OUT}/${vid}_${name}.npz" \
      --start-sec "$start" --max-sec 90 \
      --sample-interval "$si" >> "${LOG}.${vid}" 2>&1
    if [ -f "${OUT}/${vid}_${name}.npz" ]; then
      log "[$vid/$name] OK ($(stat -c%s "${OUT}/${vid}_${name}.npz") bytes)"
    else
      log "[$vid/$name][ERROR] npzが無い"
    fi
  done
  log "[$vid] 完了"
}

run_video c56 1800.0 &
run_video c60 1800.0 &
run_video c65 1800.0 &
run_video c75 1800.0 &
wait
log "=== ALL DONE ==="
