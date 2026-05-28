#!/bin/bash
# measure_stable_cell_acc.py を match_v29_02 で実行する smoke test
set -e

PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}"

mkdir -p data/verify/stable_cell_acc

echo "[measure] 開始: $(date)"
PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
  --videos match_v29_02 \
  --holdout match_v29_02 \
  --video-dir data/match_clips/v29 \
  --output data/verify/stable_cell_acc/r_case_v29m2.json \
  2>&1

echo "[measure] 完了: $(date)"
