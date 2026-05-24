#!/bin/bash
# cycle_18 = B3: cycle_14 model + bg_fp 50 + BG_FP_FORCE_MAX_PUYO 緩和
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
LOG=logs/cycle18_pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "=== START cycle18 $(date) ==="

echo "[stage1] cycle_18 viz (hsv_seed model + bg_fp 50 + force max=144)"
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
  --cycle 18 --parallel 3 \
  --cnn-model models/cnn_phase_i_hsv_seed.pt \
  --cnn-override-prob 0.70 \
  --hsv-state data/per_video_hsv_ranges/_merged_default.json \
  2>&1 | tail -30
echo "[stage1] DONE $(date)"

echo "[stage2] cycle_18 metrics"
PYTHONPATH=. ./venv/bin/python -m scripts.cycle_metrics \
  'viz_v*_multicycle_5.log' \
  'viz_v*_multicycle_14.log' \
  'viz_v*_multicycle_15.log' \
  'viz_v*_multicycle_17.log' \
  'viz_v*_multicycle_18.log' \
  > logs/cycle_18_metrics.json
echo "[stage2] DONE $(date)"
echo "=== END cycle18 $(date) ==="
