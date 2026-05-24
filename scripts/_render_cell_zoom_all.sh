#!/bin/bash
# v06+ 残 9 segment の cell_zoom 一括生成 (一時スクリプト).
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
SEGS=(
    v06_m04_445_475
    v06_m05_484_514
    v12_m03_387_417
    v12_m04_465_495
    v12_m05_538_568
    v16_m03_323_353
    v16_m04_361_391
    v16_m05_399_429
    v19_m04_445_475
)
for seg in "${SEGS[@]}"; do
    echo "== $seg =="
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_render_cell_zoom --segment "$seg" 2>&1 | tail -3
done
