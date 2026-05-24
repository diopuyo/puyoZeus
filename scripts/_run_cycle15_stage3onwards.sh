#!/bin/bash
# cycle_15 stage3+: fine-tune (with empty) → cycle_15 viz → 集計
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

LOG=logs/cycle15_stage3plus.log
exec > >(tee -a "$LOG") 2>&1
echo "=== START stage3+ $(date) ==="

OUT=data/pseudo_labels_hsv_seed_with_empty

# stage3 fine-tune
echo "[stage3] fine-tune cell_color (with empty)"
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --all \
  --store-root "$OUT" \
  --cell-base-model models/cnn_phase_b_large_v3.pt \
  --cell-save-to models/cnn_phase_i_hsv_seed_v2.pt \
  --cell-arch large \
  --class-balance \
  --augment \
  --epochs 5 \
  2>&1 | tail -100
if [ ! -f models/cnn_phase_i_hsv_seed_v2.pt ]; then
  echo "[stage3] FAILED"
  exit 1
fi
echo "[stage3] DONE $(date)"
ls -la models/cnn_phase_i_hsv_seed_v2.pt

# stage4 cycle_15 viz
echo "[stage4] cycle_15 viz"
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
  --cycle 15 --parallel 3 \
  --cnn-model models/cnn_phase_i_hsv_seed_v2.pt \
  --cnn-override-prob 0.70 \
  --hsv-state data/per_video_hsv_ranges/_merged_default.json \
  2>&1 | tail -50
echo "[stage4] DONE $(date)"

# stage5 metrics
echo "[stage5] cycle metrics"
PYTHONPATH=. ./venv/bin/python -m scripts.cycle_metrics \
  'viz_v*_multicycle_5.log' \
  'viz_v*_multicycle_14.log' \
  'viz_v*_multicycle_15.log' \
  > logs/cycle_15_metrics.json
echo "[stage5] DONE $(date)"
echo "=== END $(date) ==="
