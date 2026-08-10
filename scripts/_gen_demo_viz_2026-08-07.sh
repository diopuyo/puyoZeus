#!/bin/bash
# YouTubeデモ用: DIO vs TS match_01帯クリップの認識可視化 (新標準構成フルON)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition \
  --video data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4 \
  --output data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_recognition_viz.mp4 \
  --cnn-model models/cnn_phase_i_hsv_seed.pt \
  --enable-effect-gate --enable-burst-guard-v2 --enable-transition-merge-guard \
  --burst-gate-open-threshold 0.954 --enable-hidden-row-burst-guard \
  --enable-match-transition-debounce
echo "EXIT=$? $(date)"
ls -lh data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_recognition_viz.mp4
