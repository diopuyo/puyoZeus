#!/bin/bash
# Phase L seed 38 動画一括可視化 (= ユーザー目視用)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health phase_l_seed_review

mkdir -p data/seed_review

for d in data/phase_l/seeds/v*m*/; do
  key=$(basename "$d")
  if [ ! -f "${d}cell.jsonl" ]; then
    echo "[skip] $key (no cell.jsonl)"
    continue
  fi
  run_item visualize "$key" \
    ./venv/bin/python -m scripts.visualize_seed_samples \
      --seed-root "$d" \
      --output "data/seed_review/phase_l_${key}.png" \
      --per-color 30
done

finalize_health 0
