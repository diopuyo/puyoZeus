#!/bin/bash
# 圧力・得点タイブレークの扱いを 3 案で比較する A/B (2026-08-09)。
#   base : 従来 (おじゃま個数ベースの圧力 + 得点タイブレーク)
#   nsl  : 得点タイブレークのみ無効
#   cap  : 得点タイブレーク無効 + 圧力を「相手の盤面能力の低下量」で測る
# user 伝授:
#   「スコア差はおじゃまを送る手段で、送った時点で意味を失う」
#   「おじゃまは個数でなく、飽和連鎖量が減った・連鎖がつながりづらくなった で判断すべき」
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
OUTDIR=data/verify/youtube_demo_2026-08-07
ADV=$(./venv/bin/python -c "from src.production_config import advantage_overlay_flags; print(advantage_overlay_flags())")
echo "[config] $ADV"

CMD_CAP="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay --video $IN $ADV --no-score-lead-bias --capability-pressure --out $OUTDIR/_ab_capability_pressure.mp4"
{ echo "[cmd] $CMD_CAP"; eval "$CMD_CAP"; } > logs/_ab_cap_pressure_2026-08-09.log 2>&1
echo "AB_CAP_DONE $(date)"
ls -lh $OUTDIR/_ab_capability_pressure.mp4
