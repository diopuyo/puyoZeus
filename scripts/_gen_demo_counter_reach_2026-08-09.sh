#!/bin/bash
# 打ち合い応手確率 (モンテカルロ) を有効にした有利不利デモ (2026-08-09 user採用)。
# 相手が閾値以上を返せる確率を見て、返せない攻撃を持っている側を有利にする。
# #24 Step2 で実装・検証済みだったが有利不利には未接続だった機構を繋ぐ。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
OUTDIR=data/verify/youtube_demo_2026-08-07/release
mkdir -p $OUTDIR
ADV=$(./venv/bin/python -c "from src.production_config import advantage_overlay_flags; print(advantage_overlay_flags())")
echo "[config] ADV: $ADV --counter-reach"
CMD="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay --video $IN $ADV --counter-reach --out $OUTDIR/final_4_counter_reach.mp4"
{ echo "[cmd] $CMD"; eval "$CMD"; } > logs/final_4_counter_2026-08-09.log 2>&1
echo "COUNTER_DONE $(date)"
ls -lh $OUTDIR/final_4_counter_reach.mp4
