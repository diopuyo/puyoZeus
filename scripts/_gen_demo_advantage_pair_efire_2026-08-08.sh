#!/bin/bash
# YouTube デモ用: 同一試合 (dio vs TS m01) から 2 本を生成する。
#   映像A = 有利不利の表示のみ (視聴者が試合として見られる版)
#   映像B = A に認識オーバーレイを重ねた版 (--show-recognition)
# 台本 docs/YOUTUBE_SCRIPT_2026-08-08.md の前半/後半に対応する。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
OUTDIR=data/verify/youtube_demo_2026-08-07
# 有利不利オーバーレイは認識を毎フレーム回す必要がある (0.5s 間引きは
# おじゃま会計を壊すと docs/ADVANTAGE_OVERLAY_2026-07-13.md §2-3 に実証済み)。
COMMON="--video $IN --sample-interval 0 --platt-calibration --early-fire-reaction"

CMD_A="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay $COMMON --out $OUTDIR/demo_A_advantage_only_efire_2026-08-08.mp4"
CMD_B="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay $COMMON --show-recognition --out $OUTDIR/demo_B_advantage_with_recognition_efire_2026-08-08.mp4"

{ echo "[cmd] $CMD_A"; eval "$CMD_A"; } > logs/demo_adv_A_efire_2026-08-08.log 2>&1 &
{ echo "[cmd] $CMD_B"; eval "$CMD_B"; } > logs/demo_adv_B_efire_2026-08-08.log 2>&1 &
wait
echo "DEMO_PAIR_DONE $(date)"
ls -lh $OUTDIR/demo_A_advantage_only_efire_2026-08-08.mp4 $OUTDIR/demo_B_advantage_with_recognition_efire_2026-08-08.mp4
