#!/bin/bash
# 条件1の密displayからWIN★勝者根拠を作る。既存成果物は上書きしない。
set -euo pipefail
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

ROOT=data/verify/gate4_formal_dense_2026-08-26
for i in $(seq 1 8); do
  test -s "$ROOT/cond1_off_baseline/seg$(printf '%02d' "$i")_display.npz"
done
./venv/bin/python scripts/_extract_gate4_panel_truth_2026-08-26.py \
  --video data/frames/video_zenchi_c0BQoMJwwQU.mp4 \
  --display-dir "$ROOT/cond1_off_baseline" \
  --override-tsv "$ROOT/win_panel_truth_manual_overrides.tsv" \
  --out "$ROOT/win_panel_truth.tsv"
