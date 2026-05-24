#!/bin/bash
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
for vid in v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11; do
  echo "=== visualize ${vid} ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_seed_samples \
    --seed-root "data/pseudo_labels_hsv_seed/${vid}" \
    --output "data/seed_review/cycle32c_${vid}.png" \
    --per-color 20 2>&1 | tail -3
done
echo "=== ALL DONE ==="
ls -la data/seed_review/cycle32c_*.png
