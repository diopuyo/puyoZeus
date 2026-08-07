#!/bin/bash
# FINAL3: v3モデル (空+おじゃまseed) × 通常プレイ常時表示 × 連鎖後30フレームホールド
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
MODEL=models/cnn_finetune_olRyxDGacbg_demo_v3_2026-08-07.pt
nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition \
  --video "$IN" \
  --output data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL3_v3_hold_viz.mp4 \
  --cnn-model "$MODEL" \
  --enable-effect-gate --enable-burst-guard-v2 --enable-transition-merge-guard \
  --burst-gate-open-threshold 0.954 --enable-hidden-row-burst-guard \
  --enable-match-transition-debounce \
  --overlay-show-states stable,tsumo_fall,gravity_settle \
  --overlay-transition-hold-frames 30 \
  > logs/demo_viz_final3_2026-08-07.log 2>&1
echo "FINAL3_DONE $(date)"
ls -lh data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL3_v3_hold_viz.mp4
