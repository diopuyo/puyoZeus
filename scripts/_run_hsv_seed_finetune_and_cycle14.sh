#!/bin/bash
# HSV-seed fine-tune → cycle_14 検証 → 集計 を sequential 実行する。
# 完了後ログを `logs/hsv_seed_pipeline.log` にまとめる。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

LOG=logs/hsv_seed_pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "=== START $(date) ==="

# --- 1. fine-tune ---
echo "[stage1] fine-tune cell_color (epochs=5, class_balance + augment)"
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --all \
  --store-root data/pseudo_labels_hsv_seed_no_ojama \
  --cell-base-model models/cnn_phase_b_large_v3.pt \
  --cell-save-to models/cnn_phase_i_hsv_seed.pt \
  --cell-arch large \
  --class-balance \
  --augment \
  --epochs 5 \
  2>&1 | tail -200

if [ ! -f models/cnn_phase_i_hsv_seed.pt ]; then
  echo "[stage1] FAILED: model file not created"
  exit 1
fi
echo "[stage1] DONE $(date)"
ls -la models/cnn_phase_i_hsv_seed.pt

# --- 2. cycle_14 viz (5 videos, 3 parallel) ---
echo "[stage2] cycle_14 viz (cnn_phase_i_hsv_seed.pt, 0.70, merged_default)"
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
  --cycle 14 --parallel 3 \
  --cnn-model models/cnn_phase_i_hsv_seed.pt \
  --cnn-override-prob 0.70 \
  --hsv-state data/per_video_hsv_ranges/_merged_default.json \
  2>&1 | tail -50
echo "[stage2] DONE $(date)"

# --- 3. metrics 集計 ---
echo "[stage3] cycle metrics (cycle_5 / cycle_12 / cycle_14)"
PYTHONPATH=. ./venv/bin/python -m scripts.cycle_metrics \
  'viz_v*_multicycle_5.log' \
  'viz_v*_multicycle_12.log' \
  'viz_v*_multicycle_14.log' \
  > logs/cycle_14_metrics.json
echo "[stage3] DONE $(date)"
head -100 logs/cycle_14_metrics.json
echo "=== END $(date) ==="
