#!/bin/bash
# Codex v4: 開始設置除外・実CHAIN限定・同一交換再武装禁止・密な実表示dump。
set -euo pipefail
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

MODE="${1:?off または on を指定}"
TAG="${2:-v4}"
if [[ "$MODE" != "off" && "$MODE" != "on" ]]; then
  echo "mode must be off or on" >&2
  exit 2
fi

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

OUT=data/verify/abs_chain_end_2026-08-26
LOG=logs/abs_chain_end_2026-08-26
VIDEO=data/frames/video_zenchi_c0BQoMJwwQU.mp4
MODEL=data/verify/retrain_model62_2026-08-21
mkdir -p "$OUT" "$LOG"

EXTRA=()
if [[ "$MODE" == "on" ]]; then
  EXTRA+=(--resolved-absolute-chain-end)
fi

nice -n 19 ./venv/bin/python scripts/visualize_advantage_overlay.py \
  --video "$VIDEO" --start-sec 1738.3 --end-sec 1890.0 --warmup-sec 30 \
  --no-render --layout panel --panel-subtitle-h 0 --no-force-in-match \
  --model-dir "$MODEL" \
  --kill-override-chain-completion --enable-slide-exit-min-display-guard \
  --early-fire-reaction --per-side-settled --no-score-lead-bias --no-pressure \
  --sample-interval 0 --counter-reach --normalize-fps-30 \
  --production-recognition --resize-1080p --resolved-exchange-eval \
  --resolved-decisive-amplify --resolved-live-defender \
  --resolved-live-defender-strict --resolved-kill-override \
  --dump-timeline "$OUT/window_${TAG}_${MODE}.npz" \
  --dump-display-timeline "$OUT/window_${TAG}_${MODE}_display.npz" \
  --out "$OUT/window_${TAG}_${MODE}.mp4" "${EXTRA[@]}" \
  > "$LOG/${TAG}_${MODE}.log" 2>&1
