#!/bin/bash
# cycle_16 = A1: cycle_15 seed のまま --class-balance OFF で fine-tune し直し
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
LOG=logs/cycle16_pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "=== START cycle16 $(date) ==="

# stage1 fine-tune (no class_balance)
echo "[stage1] fine-tune cell_color (class_balance OFF)"
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --all \
  --store-root data/pseudo_labels_hsv_seed_with_empty \
  --cell-base-model models/cnn_phase_b_large_v3.pt \
  --cell-save-to models/cnn_phase_i_hsv_seed_v3.pt \
  --cell-arch large \
  --augment \
  --epochs 5 \
  2>&1 | tail -50
if [ ! -f models/cnn_phase_i_hsv_seed_v3.pt ]; then
  echo "[stage1] FAILED"
  exit 1
fi
echo "[stage1] DONE $(date)"

# stage2 cycle_16 viz
echo "[stage2] cycle_16 viz"
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
  --cycle 16 --parallel 3 \
  --cnn-model models/cnn_phase_i_hsv_seed_v3.pt \
  --cnn-override-prob 0.70 \
  --hsv-state data/per_video_hsv_ranges/_merged_default.json \
  2>&1 | tail -30
echo "[stage2] DONE $(date)"

# stage3 metrics
echo "[stage3] cycle_16 metrics"
PYTHONPATH=. ./venv/bin/python -m scripts.cycle_metrics \
  'viz_v*_multicycle_5.log' \
  'viz_v*_multicycle_14.log' \
  'viz_v*_multicycle_15.log' \
  'viz_v*_multicycle_16.log' \
  > logs/cycle_16_metrics.json
echo "[stage3] DONE $(date)"
echo "=== END cycle16 $(date) ==="
