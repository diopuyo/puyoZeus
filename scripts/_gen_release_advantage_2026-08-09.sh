#!/bin/bash
# 公開用 3本目: 有利不利のバー + 評価値グラフのみ (認識オーバーレイなし)。
# 構成は src/production_config.py から取得 (手書きしない)。
# 採用済み: 早期発火 / 片側独立更新 / 得点タイブレーク除去 / 圧力除去 / Platt較正
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
OUTDIR=data/verify/youtube_demo_2026-08-07/release
mkdir -p $OUTDIR
ADV=$(./venv/bin/python -c "from src.production_config import advantage_overlay_flags; print(advantage_overlay_flags())")
echo "[config] ADV: $ADV"
CMD="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay --video $IN $ADV --out $OUTDIR/final_3_advantage_only.mp4"
{ echo "[cmd] $CMD"; eval "$CMD"; } > logs/final_3_2026-08-09.log 2>&1
echo "FINAL3_DONE $(date)"
ls -lh $OUTDIR/final_3_advantage_only.mp4
