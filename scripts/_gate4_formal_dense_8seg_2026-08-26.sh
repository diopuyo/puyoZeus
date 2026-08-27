#!/bin/bash
# Gate 4正式検収: 全frameの実表示値を含む密timelineを8区間で収集する。
set -euo pipefail
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

COND="${1:?条件番号1-4}"
SEG_START="${2:-1}"
SEG_END="${3:-8}"
PARALLEL="${4:-3}"
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

SNAP=data/verify/gate4_formal_dense_2026-08-26/_snapshot_codex_20260826
VIDEO=data/frames/video_zenchi_c0BQoMJwwQU.mp4
MODEL=data/verify/retrain_model62_2026-08-21
BOUNDS=(0 893.7 1738.3 2637.3 3626.0 4379.5 5255.6 6131.6 7033.6)

BASEFLAGS=(
  --no-render --layout panel --panel-subtitle-h 0 --no-force-in-match
  --model-dir "$MODEL" --warmup-sec 30
  --kill-override-chain-completion --enable-slide-exit-min-display-guard
  --early-fire-reaction --per-side-settled --no-score-lead-bias --no-pressure
  --sample-interval 0 --counter-reach --normalize-fps-30
  --production-recognition --resize-1080p --resolved-exchange-eval
  --resolved-decisive-amplify --resolved-live-defender
  --resolved-live-defender-strict --resolved-kill-override
  --resolved-absolute-chain-end --death-confirm-sequence
)
case "$COND" in
  1) NAME=cond1_off_baseline; EXTRA=() ;;
  2) NAME=cond2_hysteresis_only; EXTRA=(--kill-override-hysteresis) ;;
  3) NAME=cond3_scale_compare_only; EXTRA=(--kill-override-scale-compare) ;;
  4) NAME=cond4_a_plus_b; EXTRA=(--kill-override-hysteresis --kill-override-scale-compare) ;;
  *) echo "条件は1-4。条件5は交換episode配線の独立検収後に別runnerで扱う" >&2; exit 1 ;;
esac

OUT=data/verify/gate4_formal_dense_2026-08-26/$NAME
LOG=logs/gate4_formal_dense_2026-08-26/$NAME
mkdir -p "$OUT" "$LOG"
for i in $(seq "$SEG_START" "$SEG_END"); do
  stem="seg$(printf '%02d' "$i")"
  if [[ -e "$OUT/${stem}_display.npz" || -e "$OUT/${stem}_timeline.npz" \
        || -e "$LOG/${stem}.log" ]]; then
    echo "既存成果物は上書きしない: $stem" >&2
    exit 1
  fi
done
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "snapshot=$SNAP"
  echo "condition=$COND:$NAME"
  printf 'baseflags='; printf '%q ' "${BASEFLAGS[@]}"; echo
  printf 'extra='; printf '%q ' "${EXTRA[@]}"; echo
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
    "${BASEFLAGS[@]}" "${EXTRA[@]}" \
    --dump-timeline "$OUT/${stem}_timeline.npz" \
    --dump-display-timeline "$OUT/${stem}_display.npz" \
    --out "$OUT/${stem}.mp4"
  echo "SEGMENT_DONE cond=$COND seg=$i elapsed_sec=$(($(date +%s) - t0))"
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
echo "CONDITION_DONE cond=$COND name=$NAME at=$(date --iso-8601=seconds)"
