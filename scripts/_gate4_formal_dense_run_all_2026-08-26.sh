#!/bin/bash
# user指摘の核心である規模比較を早く得るため 1→3→2→4。各条件内の並列は3。
set -euo pipefail
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
LOG=logs/gate4_formal_dense_2026-08-26/all.log
mkdir -p "$(dirname "$LOG")"
for cond in 1 3 2 4; do
  echo "CONDITION_START cond=$cond at=$(date --iso-8601=seconds)"
  bash scripts/_gate4_formal_dense_8seg_2026-08-26.sh "$cond" 1 8 3
done
echo "ALL_CONDITIONS_DONE at=$(date --iso-8601=seconds)"
