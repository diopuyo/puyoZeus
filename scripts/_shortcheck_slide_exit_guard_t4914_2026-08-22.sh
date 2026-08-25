#!/bin/bash
# もう一方の実アンカー t=4914.533 の短区間確認 (修正①+③組み合わせ、2026-08-22)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
export CV_NUM_THREADS=1

VIDEO=data/frames/video_zenchi_c0BQoMJwwQU.mp4
MODEL=data/verify/retrain_model62_2026-08-21
OUTDIR=data/verify/slide_exit_guard_shortcheck_2026-08-22
LOGDIR=logs/slide_exit_guard_shortcheck_2026-08-22
WARMUP=30
START=4884.53
END=4945.0

mkdir -p "$OUTDIR" "$LOGDIR"

ADOPTED=$(PYTHONPATH=. ./venv/bin/python -c \
  "import src.production_config as pc; print(pc.advantage_overlay_flags())")
if [ -z "$ADOPTED" ]; then
  echo "=== 中止: advantage_overlay_flags() が空文字を返した ==="
  exit 1
fi

FLAGS="--layout panel --panel-subtitle-h 0 --no-force-in-match --no-render \
--model-dir $MODEL --warmup-sec $WARMUP \
--resolved-exchange-eval --resolved-decisive-amplify --resolved-live-defender \
--kill-override-chain-completion \
--enable-slide-exit-min-display-guard \
$ADOPTED"

echo "=== 短区間確認 (t=4914.533アンカー) start $(date +%F_%T) ==="
echo "[構成] $FLAGS"

DUMP="$OUTDIR/short_t4914.npz"
LOG="$LOGDIR/run_t4914.log"

PYTHONPATH=. ./venv/bin/python -m scripts.visualize_advantage_overlay \
  --video "$VIDEO" --start-sec "$START" --end-sec "$END" \
  $FLAGS --dump-timeline "$DUMP" --out "$OUTDIR/_dummy_t4914.mp4" \
  > "$LOG" 2>&1
echo "=== rc=$? $(date +%F_%T) ==="
tail -10 "$LOG"
