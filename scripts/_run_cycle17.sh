#!/bin/bash
# cycle_17 = B: cycle_14 model (cnn_phase_i_hsv_seed.pt) + bg_fp 閾値 35→50 で viz
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
LOG=logs/cycle17_pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "=== START cycle17 $(date) ==="

# cycle_17 viz
echo "[stage1] cycle_17 viz (cnn_phase_i_hsv_seed.pt + bg_fp 50)"
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
  --cycle 17 --parallel 3 \
  --cnn-model models/cnn_phase_i_hsv_seed.pt \
  --cnn-override-prob 0.70 \
  --hsv-state data/per_video_hsv_ranges/_merged_default.json \
  2>&1 | tail -30
echo "[stage1] DONE $(date)"

# metrics
echo "[stage2] cycle_17 metrics"
PYTHONPATH=. ./venv/bin/python -m scripts.cycle_metrics \
  'viz_v*_multicycle_5.log' \
  'viz_v*_multicycle_14.log' \
  'viz_v*_multicycle_15.log' \
  'viz_v*_multicycle_17.log' \
  > logs/cycle_17_metrics.json
echo "[stage2] DONE $(date)"
echo "=== END cycle17 $(date) ==="
