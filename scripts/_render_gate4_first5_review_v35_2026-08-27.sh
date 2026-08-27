#!/bin/bash
# 条件5 v35（実盤面単発補正+極端な40点方向反転ガード）の先頭5試合レビュー動画。
set -euo pipefail
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
export CV_NUM_THREADS=1

SNAP=data/verify/gate4_condition5_2026-08-26/_snapshot_cond5_codex_20260827_v35
MODEL=data/verify/retrain_model62_2026-08-21
VIDEO=data/frames/video_zenchi_c0BQoMJwwQU.mp4
OUT=data/verify/gate4_first5_review_2026-08-27
LOG=logs/gate4_first5_review_2026-08-27
STEM=gate4_first5_cond5_v35_review_extreme_flip_guard
VIDEO_OUT="$OUT/${STEM}.mp4"
VIDEO_SILENT="$OUT/${STEM}_silent.mp4"
TIMELINE_OUT="$OUT/${STEM}_timeline.npz"
DISPLAY_OUT="$OUT/${STEM}_display.npz"
EPISODE_OUT="$OUT/${STEM}_episode.npz"
LOG_OUT="$LOG/${STEM}.log"
MANIFEST_OUT="$LOG/run_manifest_${STEM}.txt"

test -e "$SNAP/SNAPSHOT_COMPLETE"
mkdir -p "$OUT" "$LOG"
for path in "$VIDEO_OUT" "$VIDEO_SILENT" "$TIMELINE_OUT" \
  "$DISPLAY_OUT" "$EPISODE_OUT" "$LOG_OUT" "$MANIFEST_OUT"; do
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
  --resolved-minimum-prediction-guard
)

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "condition=5:v35_extreme_flip_guard"
  echo "snapshot=$SNAP"
  echo "range_sec=86.467-411.633"
  printf 'flags='; printf '%q ' "${FLAGS[@]}"; echo
} > "$MANIFEST_OUT"

nice -n 19 ./venv/bin/python "$SNAP/scripts/visualize_advantage_overlay.py" \
  --video "$VIDEO" --start-sec 86.467 --end-sec 411.633 \
  --out "$VIDEO_SILENT" "${FLAGS[@]}" \
  --dump-timeline "$TIMELINE_OUT" \
  --dump-display-timeline "$DISPLAY_OUT" \
  --dump-exchange-episode-timeline "$EPISODE_OUT" > "$LOG_OUT" 2>&1

FFMPEG=$(./venv/bin/python -c \
  'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
"$FFMPEG" -v error -stats -n -i "$VIDEO_SILENT" -ss 86.467 -i "$VIDEO" \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -shortest \
  -movflags +faststart "$VIDEO_OUT" >> "$LOG_OUT" 2>&1
"$FFMPEG" -v error -i "$VIDEO_OUT" -map 0:v:0 -map 0:a:0 \
  -t 0.1 -f null - >> "$LOG_OUT" 2>&1
rm -f -- "$VIDEO_SILENT"

echo "REVIEW_VIDEO_DONE output=$VIDEO_OUT"
