#!/bin/bash
# 片側独立更新 (--per-side-settled) の A/B。
# 両者同時 STABLE ゲートは実測で試合時間の 72.3%・最長 13.97 秒 評価を凍結させる。
# OFF (従来) と ON を同一素材・同一構成で生成して比較する。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
OUTDIR=data/verify/youtube_demo_2026-08-07
ADV_FLAGS=$(./venv/bin/python -c "from src.production_config import advantage_overlay_flags; print(advantage_overlay_flags())")
echo "[config] ADV: $ADV_FLAGS"

CMD_ON="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay --video $IN $ADV_FLAGS --per-side-settled --out $OUTDIR/_ab_perside_on.mp4"

{ echo "[cmd] $CMD_ON"; eval "$CMD_ON"; } > logs/_ab_perside_on_2026-08-08.log 2>&1
echo "AB_DONE $(date)"
