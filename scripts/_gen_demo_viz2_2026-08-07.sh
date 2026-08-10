#!/bin/bash
# YouTubeデモ用本番2本: ウォームアップクリップ (試合1〜7、370秒) の認識可視化
#   A: 通常版 (全状態オーバーレイ)  B: STABLE時のみ表示版
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_warmup_clip.mp4
FLAGS="--cnn-model models/cnn_phase_i_hsv_seed.pt --enable-effect-gate --enable-burst-guard-v2 --enable-transition-merge-guard --burst-gate-open-threshold 0.954 --enable-hidden-row-burst-guard --enable-match-transition-debounce"
nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition \
  --video "$IN" \
  --output data/verify/youtube_demo_2026-08-07/dio_vs_ts_full_overlay_viz.mp4 \
  $FLAGS > logs/demo_viz_full_2026-08-07.log 2>&1 &
nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition \
  --video "$IN" \
  --output data/verify/youtube_demo_2026-08-07/dio_vs_ts_stable_only_viz.mp4 \
  $FLAGS --overlay-stable-only > logs/demo_viz_stable_only_2026-08-07.log 2>&1 &
wait
echo "BOTH_DONE $(date)"
ls -lh data/verify/youtube_demo_2026-08-07/dio_vs_ts_full_overlay_viz.mp4 \
       data/verify/youtube_demo_2026-08-07/dio_vs_ts_stable_only_viz.mp4
