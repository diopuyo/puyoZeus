#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONPATH=.
nice -n 10 ./venv/bin/python -u -m scripts.train_board_cnn \
  --pairs data/indicators_v2/board_pairs_fixed.npz \
  --out models/board_cnn_s4_d03_w05.pt \
  --epochs 50 --dropout 0.3 --width-mult 0.5 --flip-aug \
  --weight-decay 1e-3 --patience 8 --seed 0 --device cuda > logs/train_s4.log 2>&1
echo "EXIT=$?" >> logs/train_s4.log
