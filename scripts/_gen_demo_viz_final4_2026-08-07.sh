#!/bin/bash
# FINAL4: v3モデル × セル安定フィルタ14f × (A:全状態表示 / B:stable+ツモ落下のみ) の2本
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
MODEL=models/cnn_finetune_olRyxDGacbg_demo_v3_2026-08-07.pt
BASEFLAGS="--cnn-model $MODEL --enable-effect-gate --enable-burst-guard-v2 --enable-transition-merge-guard --burst-gate-open-threshold 0.954 --enable-hidden-row-burst-guard --enable-match-transition-debounce --overlay-cell-stability-frames 14"
nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition \
  --video "$IN" \
  --output data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL4a_all_states_viz.mp4 \
  $BASEFLAGS \
  > logs/demo_viz_final4a_2026-08-07.log 2>&1 &
nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition \
  --video "$IN" \
  --output data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL4b_stable_tsumo_viz.mp4 \
  $BASEFLAGS --overlay-show-states stable,tsumo_fall --overlay-state-debounce-frames 10 \
  > logs/demo_viz_final4b_2026-08-07.log 2>&1 &
wait
echo "FINAL4_BOTH_DONE $(date)"
ls -lh data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL4*.mp4
