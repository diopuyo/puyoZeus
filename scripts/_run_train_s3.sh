#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
nice -n 10 ./venv/bin/python -u -m scripts.train_board_cnn \
  --pairs data/indicators_v2/board_pairs_fixed.npz \
  --epochs 20 --out models/board_cnn_s3.pt > logs/train_s3.log 2>&1
echo "EXIT=$?" >> logs/train_s3.log
