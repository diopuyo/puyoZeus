#!/bin/bash
# 方針(a) 終局イベント合成込みラベル生成 (2026-08-03 main発注)。
# 既存ファイルは上書きしない (新ファイル名)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

NPZ_DIR="data/indicators_v2/boards_lean_regen_2026-07-31"
BASE_CSV="data/indicators_v2/exchange_labels_regen_synth_2026-08-03.csv"
AUG_CSV="data/indicators_v2/exchange_labels_regen_synth_aug_2026-08-03.csv"

echo "[1/2] label_exchange_outcome.py --synthesize-terminal-events 開始 $(date)"
PYTHONPATH=. ./venv/bin/python -m scripts.label_exchange_outcome \
  --npz-dir "$NPZ_DIR" --synthesize-terminal-events --output "$BASE_CSV"
echo "[1/2] 完了 $(date)"

echo "[2/2] augment_exchange_labels_with_sim.py (--workers 8) 開始 $(date)"
PYTHONPATH=. ./venv/bin/python -m scripts.augment_exchange_labels_with_sim \
  --input-csv "$BASE_CSV" --npz-dir "$NPZ_DIR" --output "$AUG_CSV" --workers 8
echo "[2/2] 完了 $(date)"
echo "[all done] $(date)"
