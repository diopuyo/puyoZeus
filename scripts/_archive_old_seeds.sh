#!/bin/bash
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
mkdir -p data/pseudo_labels_hsv_seed_archive
for d in v29_old_noisy v29_raw_partial v29m2_g1_only v29m2_g9g10_with_ojama v50 v70 v89m3_old_1779168036 v91 v97; do
  if [ -d "data/pseudo_labels_hsv_seed/$d" ]; then
    mv "data/pseudo_labels_hsv_seed/$d" "data/pseudo_labels_hsv_seed_archive/"
    echo "[archived] $d"
  fi
done
echo "---REMAINING---"
ls data/pseudo_labels_hsv_seed/ | grep -v "\.png"
