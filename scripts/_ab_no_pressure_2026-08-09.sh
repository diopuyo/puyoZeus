#!/bin/bash
# 圧力成分を完全に外した版のデモ (2026-08-09 user要望)。
# 圧力は「攻撃を通した履歴」だが、その効果は既に相手の盤面 (おじゃま数・連結・
# 飽和連鎖量) としてモデルが見ている。本当に独立した情報なのかを確かめる。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
OUTDIR=data/verify/youtube_demo_2026-08-07
ADV=$(./venv/bin/python -c "from src.production_config import advantage_overlay_flags; print(advantage_overlay_flags())")
echo "[config] $ADV"
CMD="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay --video $IN $ADV --no-score-lead-bias --no-pressure --out $OUTDIR/demo_no_pressure.mp4"
{ echo "[cmd] $CMD"; eval "$CMD"; } > logs/_ab_no_pressure_2026-08-09.log 2>&1
echo "NO_PRESSURE_DONE $(date)"
ls -lh $OUTDIR/demo_no_pressure.mp4
