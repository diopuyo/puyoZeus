#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
for v in c1 c2 c3 c4 c6 c7 c8 c9 c32 c33 c82 c84 c95; do
  ls -la "data/indicators_v2/boards_lean_regen_2026-07-31/${v}.npz"
done
