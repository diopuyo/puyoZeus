#!/bin/bash
# 条件1を基準に条件3/2/4を密なdisplay timelineで一括比較する。
set -euo pipefail
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

ROOT=data/verify/gate4_formal_dense_2026-08-26
OFF="$ROOT/cond1_off_baseline"
LOG=logs/gate4_formal_dense_2026-08-26
TRUTH="$ROOT/win_panel_truth.tsv"
mkdir -p "$LOG"
for name in cond3_scale_compare_only cond2_hysteresis_only cond4_a_plus_b; do
  ./venv/bin/python scripts/_analyze_pm100_display_pair_2026-08-26.py \
    "$OFF" "$ROOT/$name" "$TRUTH" | tee "$LOG/${name}_vs_off.log"
done
