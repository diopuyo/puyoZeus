#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
nice -n 15 ./venv/bin/python -u -m scripts.collect_boards_lean \
  --video data/frames/video_c1.mp4 \
  --out-npz data/indicators_v2/boards_lean_fixed/c1.npz \
  --sample-interval 0.2 --max-sec 900 > logs/verify_lean_label.log 2>&1
echo "EXIT=$?" >> logs/verify_lean_label.log
