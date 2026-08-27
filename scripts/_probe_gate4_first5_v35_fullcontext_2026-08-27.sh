#!/bin/bash
# v35を動画先頭から3:51後まで通し、差分範囲を描画なしで最終検証する。
set -euo pipefail
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1

SNAP=data/verify/gate4_condition5_2026-08-26/_snapshot_cond5_codex_20260827_v35
MODEL=data/verify/retrain_model62_2026-08-21
VIDEO=data/frames/video_zenchi_c0BQoMJwwQU.mp4
OUT=data/verify/gate4_first5_review_2026-08-27
LOG=logs/gate4_first5_review_2026-08-27
STEM=gate4_first5_cond5_v35_extreme_flip_guard_fullcontext
TIMELINE_OUT="$OUT/${STEM}_timeline.npz"
DISPLAY_OUT="$OUT/${STEM}_display.npz"
EPISODE_OUT="$OUT/${STEM}_episode.npz"
LOG_OUT="$LOG/${STEM}.log"

test -e "$SNAP/SNAPSHOT_COMPLETE"
mkdir -p "$OUT" "$LOG"
for path in "$TIMELINE_OUT" "$DISPLAY_OUT" "$EPISODE_OUT" "$LOG_OUT"; do
  if [[ -e "$path" ]]; then
    echo "既存成果物は上書きしない: $path" >&2
    exit 1
  fi
done

FLAGS=(
  --layout panel --panel-subtitle-h 0 --show-recognition --show-chain-count
  --no-force-in-match --model-dir "$MODEL" --warmup-sec 30
  --enable-slide-exit-min-display-guard --early-fire-reaction
  --per-side-settled --no-score-lead-bias --no-pressure
  --sample-interval 0 --counter-reach --normalize-fps-30
  --production-recognition --resize-1080p --resolved-exchange-eval
  --resolved-decisive-amplify --resolved-live-defender
  --resolved-live-defender-strict --resolved-kill-override
  --resolved-absolute-chain-end --death-confirm-sequence
  --exchange-episode-gate --gross-ledger-dump
  --resolved-episode-physical-consistency-guard
  --resolved-minimum-prediction-guard --no-render
)

nice -n 19 ./venv/bin/python "$SNAP/scripts/visualize_advantage_overlay.py" \
  --video "$VIDEO" --start-sec 86.467 --end-sec 330 \
  --out "$OUT/${STEM}_unused.mp4" "${FLAGS[@]}" \
  --dump-timeline "$TIMELINE_OUT" \
  --dump-display-timeline "$DISPLAY_OUT" \
  --dump-exchange-episode-timeline "$EPISODE_OUT" > "$LOG_OUT" 2>&1

echo "FULLCONTEXT_DONE display=$DISPLAY_OUT"
