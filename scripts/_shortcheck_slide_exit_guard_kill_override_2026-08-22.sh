#!/bin/bash
# 修正③(スライド誤検知抑制ガード) + 修正①(kill_override連鎖完走後是正) の
# 正しい組み合わせを、全編再走査の前に短区間 (t=6717.5含む) で確認する
# (2026-08-22、coordinator指摘: 組み合わせないと効かない)。
#
# 前回の inline bash -c 実行はネストしたクォート層で $ADOPTED が空展開になり
# warmup/sample-interval等が無効なまま走っていた (MSYS/WSL 経由の既知の
# escape事故、feedback_msys_pipe_escape.md)。スクリプトファイル化して回避する。
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
START=6664.17
END=6725.0

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

echo "=== 短区間確認 (修正①+③組み合わせ) start $(date +%F_%T) ==="
echo "[単一情報源] $ADOPTED"
echo "[構成] $FLAGS"
echo "[累積フラグ確認] --kill-override-chain-gen-accumulate は含まれていないこと:"
echo "$FLAGS" | grep -o "kill-override-chain-gen-accumulate" && echo "!!! 累積フラグが混入 !!!" || echo "OK: 累積フラグなし"

DUMP="$OUTDIR/short_t6717_v2.npz"
LOG="$LOGDIR/run_v2.log"

PYTHONPATH=. ./venv/bin/python -m scripts.visualize_advantage_overlay \
  --video "$VIDEO" --start-sec "$START" --end-sec "$END" \
  $FLAGS --dump-timeline "$DUMP" --out "$OUTDIR/_dummy_v2.mp4" \
  > "$LOG" 2>&1
echo "=== rc=$? $(date +%F_%T) ==="
tail -20 "$LOG"
