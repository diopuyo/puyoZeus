#!/bin/bash
# 最終デモ (2026-08-13): Rust native反撃計算 + 30fps正規化 + 0.5秒間引き入り。
# 区間 = 1試合目の実開始 (t=167秒実測) の少し前 〜 5試合目終了 (410秒)。
# 冒頭のメニュー区間はカット (user指摘 2026-08-13)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/frames/review_demo_2026-08-12.mp4
OUTDIR=data/verify/demo_3match_2026-08-12
mkdir -p $OUTDIR logs
CMD="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay \
  --video $IN \
  --early-fire-reaction --per-side-settled --no-score-lead-bias --no-pressure \
  --sample-interval 0 --layout panel --show-recognition \
  --start-sec 162 --end-sec 410 \
  --out $OUTDIR/demo_final_5match_native.mp4"
echo "[cmd] $CMD"
eval "$CMD"
echo "DEMO_FINAL_DONE $(date)"
ls -lh $OUTDIR/
