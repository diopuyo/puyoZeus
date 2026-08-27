#!/bin/bash
# Gate 3R-6候補のレビュー動画: user指定どおり先頭5試合を開始〜終了まで。
set -euo pipefail
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

OUT=data/verify/gate3r6_review_codex_2026-08-26
LOG=logs/gate3r6_review_codex_2026-08-26
mkdir -p "$OUT" "$LOG"

nice -n 19 ./venv/bin/python scripts/visualize_advantage_overlay.py \
  --video data/frames/video_zenchi_c0BQoMJwwQU.mp4 \
  --out "$OUT/gate3r6_first5_review.mp4" --end-sec 420 \
  --model-dir data/verify/retrain_model62_2026-08-21 \
  --layout panel --panel-subtitle-h 0 --show-recognition --show-chain-count \
  --no-force-in-match --production-recognition --resize-1080p --normalize-fps-30 \
  --kill-override-chain-completion --enable-slide-exit-min-display-guard \
  --early-fire-reaction --per-side-settled --no-score-lead-bias --no-pressure \
  --sample-interval 0 --counter-reach --resolved-exchange-eval \
  --resolved-decisive-amplify --resolved-live-defender \
  --resolved-live-defender-strict --resolved-kill-override \
  --resolved-absolute-chain-end --death-confirm-sequence \
  --dump-timeline "$OUT/gate3r6_first5_review_timeline.npz" \
  --dump-display-timeline "$OUT/gate3r6_first5_review_display.npz" \
  > "$LOG/render.log" 2>&1
