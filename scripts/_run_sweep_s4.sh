#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONPATH=.
# 正則化強度スイープ: (width_mult, dropout) を変えて中盤回復点を探す。flip常時ON。
run() {
  local w=$1 d=$2 tag=$3
  echo "===== SWEEP tag=$tag width=$w dropout=$d ====="
  nice -n 10 ./venv/bin/python -u -m scripts.train_board_cnn \
    --pairs data/indicators_v2/board_pairs_fixed.npz \
    --out "models/board_cnn_$tag.pt" \
    --epochs 50 --dropout "$d" --width-mult "$w" --flip-aug \
    --weight-decay 1e-3 --patience 8 --seed 0 --device cuda 2>&1 \
    | grep -E "設定|中盤|序盤|終盤|全体|ベストエポック"
}
run 1.0 0.30 w10_d03
run 1.0 0.15 w10_d015
run 0.75 0.20 w075_d02
echo "SWEEP DONE"
