#!/bin/bash
# cycle_19 = B-AND: cycle_18 + ImageReader 1st pass で AND 条件
# (bg_fp 距離 < 50 AND HSV-単独でも puyo 色判定されない) で empty 確定
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
LOG=logs/cycle19_pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "=== START cycle19 $(date) ==="

echo "[stage1] cycle_19 viz"
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
  --cycle 19 --parallel 3 \
  --cnn-model models/cnn_phase_i_hsv_seed.pt \
  --cnn-override-prob 0.70 \
  --hsv-state data/per_video_hsv_ranges/_merged_default.json \
  2>&1 | tail -30
echo "[stage1] DONE $(date)"

echo "[stage2] cycle_19 metrics"
PYTHONPATH=. ./venv/bin/python -m scripts.cycle_metrics \
  'viz_v*_multicycle_5.log' \
  'viz_v*_multicycle_14.log' \
  'viz_v*_multicycle_18.log' \
  'viz_v*_multicycle_19.log' \
  > logs/cycle_19_metrics.json
echo "[stage2] DONE $(date)"
echo "=== END cycle19 $(date) ==="
