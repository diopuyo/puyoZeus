#!/bin/bash
# 得点タイブレーク無効化 (--no-score-lead-bias) の A/B。
# user 伝授「スコア差そのものに意味はない。スコアはおじゃまを送る手段で、
# 送った時点で意味を失う。評価は予告おじゃま+フィールド状況で見るのが正しい」
# を受けた検証。 t=29 (1P優位の盤面が2P有利表示になる) と、
# 「同時連鎖なのに一方へ寄り続ける」症状が消えるかを見る。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
OUTDIR=data/verify/youtube_demo_2026-08-07
ADV_FLAGS=$(./venv/bin/python -c "from src.production_config import advantage_overlay_flags; print(advantage_overlay_flags())")
echo "[config] ADV: $ADV_FLAGS"

CMD="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay --video $IN $ADV_FLAGS --no-score-lead-bias --out $OUTDIR/_ab_no_score_lead.mp4"
{ echo "[cmd] $CMD"; eval "$CMD"; } > logs/_ab_no_score_lead_2026-08-09.log 2>&1
echo "AB_DONE $(date)"
ls -lh $OUTDIR/_ab_no_score_lead.mp4
