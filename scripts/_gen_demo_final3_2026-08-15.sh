#!/bin/bash
# 最終デモ (2026-08-15): 指摘13 (--resolved-live-defender) 込みの全修正フラグON。
# 区間・他フラグは _gen_demo_fixed_2026-08-13.sh と同一 (差分=指摘13フラグのみ)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/frames/review_demo_2026-08-12.mp4
OUTDIR=data/verify/demo_fixed_2026-08-13
mkdir -p $OUTDIR logs
nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay \
  --video $IN \
  --early-fire-reaction --per-side-settled --no-score-lead-bias --no-pressure \
  --sample-interval 0 --layout panel --show-recognition \
  --counter-remaining-time --counter-defender-only \
  --stable-majority-window \
  --enable-ojama-fall-placement-override --enable-ojama-fall-entry-hardening \
  --enable-ojama-fall-scoped-exit --resolved-exchange-eval --resolved-decisive-amplify \
  --enable-pseudo-chain-score-fill --resolved-live-defender \
  --start-sec 162 --end-sec 310 \
  --out $OUTDIR/demo_final3_3match.mp4
echo "DEMO_FINAL3_DONE $(date)"
