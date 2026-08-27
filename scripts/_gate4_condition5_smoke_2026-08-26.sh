#!/bin/bash
# seg01 game2の誤反転窓を含む条件5の短時間実データsmoke。
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

TAG="${1:-gate4_condition5_smoke_2026-08-26}"
END_SEC="${2:-240}"
OUT="data/verify/$TAG"
LOG="logs/$TAG"
if [[ -e "$OUT" || -e "$LOG" ]]; then
  echo "既存smoke成果物は上書きしない: $OUT / $LOG" >&2
  exit 1
fi
mkdir -p "$OUT" "$LOG"
nice -n 19 ./venv/bin/python scripts/visualize_advantage_overlay.py \
  --video data/frames/video_zenchi_c0BQoMJwwQU.mp4 \
  --start-sec 180 --end-sec "$END_SEC" --warmup-sec 30 \
  --no-render --layout panel --panel-subtitle-h 0 --no-force-in-match \
  --model-dir data/verify/retrain_model62_2026-08-21 \
  --enable-slide-exit-min-display-guard --early-fire-reaction \
  --per-side-settled --no-score-lead-bias --no-pressure \
  --sample-interval 0 --counter-reach --normalize-fps-30 \
  --production-recognition --resize-1080p --resolved-exchange-eval \
  --resolved-decisive-amplify --resolved-live-defender \
  --resolved-live-defender-strict --resolved-kill-override \
  --resolved-absolute-chain-end --death-confirm-sequence \
  --exchange-episode-gate --gross-ledger-dump \
  --dump-timeline "$OUT/seg01_timeline.npz" \
  --dump-display-timeline "$OUT/seg01_display.npz" \
  --dump-exchange-episode-timeline "$OUT/seg01_episode.npz" \
  --out "$OUT/seg01.mp4" > "$LOG/seg01.log" 2>&1
./venv/bin/python scripts/_verify_gate4_condition5_2026-08-26.py "$OUT" 1 \
  | tee "$LOG/verify.log"
