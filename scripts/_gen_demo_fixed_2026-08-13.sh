#!/bin/bash
# 改修後デモ1本目 (2026-08-13): レビュー8件対応の全修正フラグON + 本番認識構成 (自動適用)。
# 区間 = 同一動画 (review_demo) の1〜3試合目 (source 162〜310秒、3試合で十分のuser指示)。
# 学習構成は改修前と同一 (light63 CSV) — 修正フラグだけの差分比較にするため。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/frames/review_demo_2026-08-12.mp4
OUTDIR=data/verify/demo_fixed_2026-08-13
mkdir -p $OUTDIR logs
CMD="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay \
  --video $IN \
  --early-fire-reaction --per-side-settled --no-score-lead-bias --no-pressure \
  --sample-interval 0 --layout panel --show-recognition \
  --counter-remaining-time --counter-defender-only \
  --stable-majority-window \
  --enable-ojama-fall-placement-override --enable-ojama-fall-entry-hardening \
  --enable-ojama-fall-scoped-exit --resolved-exchange-eval \
  --start-sec 162 --end-sec 310 \
  --out $OUTDIR/demo_fixed_3match.mp4"
echo "[cmd] $CMD"
eval "$CMD"
echo "DEMO_FIXED_DONE $(date)"
ls -lh $OUTDIR/
