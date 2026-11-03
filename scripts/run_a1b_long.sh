#!/bin/bash
# 長い記録を同じ資源制限・記録ロックの下で先行処理する。
set -eu
cd /mnt/d/puyo_analyzer/wt_a1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
OUT=$PWD/logs/a1b
source scripts/a1_resource_limits.sh
for mode in single combo; do
  for name in s1 s2; do
    run_limited "$OUT/long_${mode}_${name}.log" nice -n 10 "$PY" -B -m scripts.a1b_replay "$mode" "$name"
  done
done
