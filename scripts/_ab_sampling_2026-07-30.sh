#!/bin/bash
# 間引き(--sample-interval)がSTABLE/CHAIN状態機械の判定を変えるかのA/B (2026-07-30)
#
# 発見: collect_boards_lean の --sample-interval 0.2 は「fps*0.2フレームに1回だけ
# pipeline.update を呼び、他はcontinueでスキップ」= 30fps動画で実質5fps。
# 一方 visualize_advantage_overlay のレンダは全フレーム(30fps)を処理する。
# 状態機械に与えるフレーム列が6倍違うので、CHAIN遷移を取りこぼしてmid-chainの盤面を
# STABLEと誤判定し「列消失」に見える可能性がある。
#
# 検証: 開始時刻を揃え間引きの有無のみを変える。c60 の t=1467.4 付近(2P col3が20.8秒消失と
# 報告された箇所)を含む 1398〜1508秒で比較する。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
OUT=data/indicators_v2/sampling_ab_2026-07-30
LOG=logs/ab_sampling_2026-07-30.log
log() { echo "[ab-sampling] $(date) $*" >> "$LOG"; }

log "=== 開始 ==="
for spec in "sampled|0.2" "allframes|0"; do
  name="${spec%%|*}"; si="${spec##*|}"
  log "[$name] sample-interval=$si"
  nice -n 15 ./venv/bin/python -u -m scripts._collect_lean_1t \
    --video data/frames/video_c60.mp4 \
    --out-npz "${OUT}/c60_${name}.npz" \
    --start-sec 1398.0 --max-sec 110 \
    --sample-interval "$si" >> "$LOG" 2>&1
  if [ -f "${OUT}/c60_${name}.npz" ]; then
    log "[$name] OK ($(stat -c%s "${OUT}/c60_${name}.npz") bytes)"
  else
    log "[$name][ERROR] npzが無い"
  fi
done
log "=== ALL DONE ==="
