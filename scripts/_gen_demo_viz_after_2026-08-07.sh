#!/bin/bash
# 学習後デモ: fine-tune済みモデルで80秒クリップ (match_01帯) を2表示モード同時生成
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
MODEL=models/cnn_finetune_olRyxDGacbg_demo_2026-08-07.pt
FLAGS="--cnn-model $MODEL --enable-effect-gate --enable-burst-guard-v2 --enable-transition-merge-guard --burst-gate-open-threshold 0.954 --enable-hidden-row-burst-guard --enable-match-transition-debounce"
nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition \
  --video "$IN" \
  --output data/verify/youtube_demo_2026-08-07/dio_vs_ts_AFTER_full_viz.mp4 \
  $FLAGS > logs/demo_viz_after_full_2026-08-07.log 2>&1 &
nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition \
  --video "$IN" \
  --output data/verify/youtube_demo_2026-08-07/dio_vs_ts_AFTER_stable_only_viz.mp4 \
  $FLAGS --overlay-stable-only > logs/demo_viz_after_stable_2026-08-07.log 2>&1 &
wait
echo "AFTER_BOTH_DONE $(date)"
ls -lh data/verify/youtube_demo_2026-08-07/dio_vs_ts_AFTER_*.mp4
