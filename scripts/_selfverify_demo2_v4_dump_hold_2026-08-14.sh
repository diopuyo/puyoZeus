#!/bin/bash
# 検収セルフベリファイ (demo2 v4) ホールド区間限定版: 出力t=34-38秒
# (source t=264-268s) をカバーする範囲だけを --no-render + --dump-timeline で
# 高速取得する。warmup 60秒 (試合開始t=230からの内部状態構築のため十分な余裕)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/frames/video_74.mp4
OUTDIR=data/verify/demo_fixed_2026-08-13
mkdir -p $OUTDIR logs
CMD="./venv/bin/python -u -m scripts.visualize_advantage_overlay \
  --video $IN \
  --early-fire-reaction --per-side-settled --no-score-lead-bias --no-pressure \
  --sample-interval 0 --layout panel --show-recognition \
  --counter-remaining-time --counter-defender-only \
  --stable-majority-window \
  --enable-ojama-fall-placement-override --enable-ojama-fall-entry-hardening \
  --enable-ojama-fall-scoped-exit --resolved-exchange-eval --enable-pseudo-chain-score-fill \
  --start-sec 230 --end-sec 270 \
  --no-render --dump-timeline $OUTDIR/demo2_v4_selfverify_hold_dump.npz \
  --out $OUTDIR/_unused_selfverify.mp4"
echo "[cmd] $CMD"
eval "$CMD"
echo "SELFVERIFY_HOLD_DUMP_DONE $(date)"
