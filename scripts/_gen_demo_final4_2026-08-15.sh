#!/bin/bash
# 最終デモ final4 (2026-08-15): 指摘14修正 (採用登録済み2フラグ) + れんさ数表示。
#
# final3 からの差分:
#   + --resolved-live-defender-strict  (指摘14 案1、user承認8/15、採用登録済み)
#   + --resolved-kill-override         (指摘14 案2、user承認8/15、採用登録済み)
#   + --show-chain-count               (れんさ数表示、user要望8/15、既定OFFの新機能)
#   - --stable-majority-window         (8/15 不採用確定 = 効果ゼロ)
#   - --enable-ojama-fall-entry-hardening (8/15 不採用確定 = 悪化主犯)
#   - --enable-ojama-fall-scoped-exit  (8/15 不採用確定 = 寄与ゼロ)
#
# 注意 (2026-08-15 判明): 指摘9〜13の修正フラグ群 (resolved-exchange-eval /
# resolved-decisive-amplify / resolved-live-defender / pseudo-chain-score-fill /
# counter-remaining-time / counter-defender-only) は userレビュー合格済みだが
# production_config.py に未登録・CLI既定OFF のため、ここで明示指定が必須。
# この管理上の穴は user へ報告済み (採用登録の要否は user 判断)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/frames/review_demo_2026-08-12.mp4
OUTDIR=data/verify/demo_final4_2026-08-15
mkdir -p $OUTDIR logs
nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay \
  --video $IN \
  --early-fire-reaction --per-side-settled --no-score-lead-bias --no-pressure \
  --sample-interval 0 --layout panel --show-recognition \
  --counter-remaining-time --counter-defender-only \
  --enable-ojama-fall-placement-override \
  --resolved-exchange-eval --resolved-decisive-amplify \
  --enable-pseudo-chain-score-fill --resolved-live-defender \
  --resolved-live-defender-strict --resolved-kill-override \
  --show-chain-count \
  --start-sec 162 --end-sec 310 \
  --out $OUTDIR/demo_final4_3match.mp4
echo "DEMO_FINAL4_DONE $(date)"
