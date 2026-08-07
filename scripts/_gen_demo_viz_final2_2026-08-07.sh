#!/bin/bash
# 最終版デモ: v2モデル × (STABLE+ツモ落下表示 / 全状態表示) の2本
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
MODEL=models/cnn_finetune_olRyxDGacbg_demo_v2_2026-08-07.pt
FLAGS="--cnn-model $MODEL --enable-effect-gate --enable-burst-guard-v2 --enable-transition-merge-guard --burst-gate-open-threshold 0.954 --enable-hidden-row-burst-guard --enable-match-transition-debounce"
nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition \
  --video "$IN" \
  --output data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL2_play_visible_viz.mp4 \
  $FLAGS --overlay-show-states stable,tsumo_fall,gravity_settle \
  > logs/demo_viz_final2_2026-08-07.log 2>&1 &
wait
echo "FINAL2_DONE $(date)"
ls -lh data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL2_*.mp4
