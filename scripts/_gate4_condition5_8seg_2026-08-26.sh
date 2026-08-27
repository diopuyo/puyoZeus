#!/bin/bash
# 条件5: 交換台帳 + 未解決hard overrideゲート + chain_id置換会計。
set -euo pipefail
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

SEG_START="${1:-1}"
SEG_END="${2:-8}"
PARALLEL="${3:-3}"
if [[ "$PARALLEL" -gt 3 ]]; then
  echo "並列上限は3" >&2
  exit 1
fi

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
export CV_NUM_THREADS=1

SNAP=data/verify/gate4_condition5_2026-08-26/_snapshot_cond5_codex_20260827_v12
VIDEO=data/frames/video_zenchi_c0BQoMJwwQU.mp4
MODEL=data/verify/retrain_model62_2026-08-21
OUT=data/verify/gate4_condition5_2026-08-26/cond5_exchange_episode_v12
LOG=logs/gate4_condition5_2026-08-26/cond5_exchange_episode_v12
BOUNDS=(0 893.7 1738.3 2637.3 3626.0 4379.5 5255.6 6131.6 7033.6)
BASEFLAGS=(
  --no-render --layout panel --panel-subtitle-h 0 --no-force-in-match
  --model-dir "$MODEL" --warmup-sec 30
  --enable-slide-exit-min-display-guard --early-fire-reaction
  --per-side-settled --no-score-lead-bias --no-pressure
  --sample-interval 0 --counter-reach --normalize-fps-30
  --production-recognition --resize-1080p --resolved-exchange-eval
  --resolved-decisive-amplify --resolved-live-defender
  --resolved-live-defender-strict --resolved-kill-override
  --resolved-absolute-chain-end --death-confirm-sequence
  --exchange-episode-gate --gross-ledger-dump
)

mkdir -p "$OUT" "$LOG"
if [[ ! -e "$SNAP/SNAPSHOT_COMPLETE" ]]; then
  echo "条件5snapshotが未作成: $SNAP" >&2
  exit 1
fi
for i in $(seq "$SEG_START" "$SEG_END"); do
  stem="seg$(printf '%02d' "$i")"
  if [[ -e "$OUT/${stem}_episode.npz" || -e "$LOG/${stem}.log" ]]; then
    echo "既存成果物は上書きしない: $stem" >&2
    exit 1
  fi
done
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "snapshot=$SNAP"
  echo "condition=5:cond5_exchange_episode"
  printf 'baseflags='; printf '%q ' "${BASEFLAGS[@]}"; echo
  echo "old_chain_generation_accumulator=OFF"
  echo "segments=$SEG_START-$SEG_END parallel=$PARALLEL"
} > "$LOG/run_manifest_${SEG_START}_${SEG_END}.txt"

run_segment() {
  local i="$1" s e stem t0
  s="${BOUNDS[$((i - 1))]}"
  e="${BOUNDS[$i]}"
  stem="seg$(printf '%02d' "$i")"
  t0=$(date +%s)
  nice -n 19 ./venv/bin/python "$SNAP/scripts/visualize_advantage_overlay.py" \
    --video "$VIDEO" --start-sec "$s" --end-sec "$e" \
    "${BASEFLAGS[@]}" \
    --dump-timeline "$OUT/${stem}_timeline.npz" \
    --dump-display-timeline "$OUT/${stem}_display.npz" \
    --dump-exchange-episode-timeline "$OUT/${stem}_episode.npz" \
    --out "$OUT/${stem}.mp4"
  echo "SEGMENT_DONE cond=5 seg=$i elapsed_sec=$(($(date +%s) - t0))"
}

running=0
for i in $(seq "$SEG_START" "$SEG_END"); do
  run_segment "$i" > "$LOG/seg$(printf '%02d' "$i").log" 2>&1 &
  running=$((running + 1))
  if [[ "$running" -ge "$PARALLEL" ]]; then
    wait -n
    running=$((running - 1))
  fi
done
wait
echo "CONDITION_DONE cond=5 at=$(date --iso-8601=seconds)"
