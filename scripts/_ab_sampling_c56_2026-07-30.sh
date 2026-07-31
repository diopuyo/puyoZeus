#!/bin/bash
# c56 での間引きA/B (2026-07-30)。c60 では間引きの有無で結果が変わらなかったが n=1 だった。
# c56 2P col1 (t=2691.4) は間引きnpzが「col1完全に空」と記録する一方、
# 全フレーム処理のレンダ表示ではcol1にラベルが付いており食い違う。
# 開始時刻を揃え間引きの有無のみを変えて、どちらが正しいかを決める。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
OUT=data/indicators_v2/sampling_ab_2026-07-30
LOG=logs/ab_sampling_c56_2026-07-30.log
log() { echo "[ab-c56] $(date) $*" >> "$LOG"; }
log "=== 開始 ==="
for spec in "sampled|0.2" "allframes|0"; do
  name="${spec%%|*}"; si="${spec##*|}"
  log "[$name] sample-interval=$si"
  nice -n 15 ./venv/bin/python -u -m scripts._collect_lean_1t \
    --video data/frames/video_c56.mp4 \
    --out-npz "${OUT}/c56_${name}.npz" \
    --start-sec 2640.0 --max-sec 80 \
    --sample-interval "$si" >> "$LOG" 2>&1
  if [ -f "${OUT}/c56_${name}.npz" ]; then
    log "[$name] OK ($(stat -c%s "${OUT}/c56_${name}.npz") bytes)"
  else
    log "[$name][ERROR] npzが無い"
  fi
done
log "=== ALL DONE ==="
