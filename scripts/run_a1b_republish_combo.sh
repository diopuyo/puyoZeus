#!/bin/bash
# 使用側への再公開修正後、併用18記録を全件作り直す。
set -eu
cd /mnt/d/puyo_analyzer/wt_a1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
OUT=$PWD/logs/a1b
source scripts/a1_resource_limits.sh
case "$1" in
  0) names='review q_7gc4TgFig mia8KCjr52g c2 c5 s0 s3';;
  1) names='zenchi c3 c6 s1 s4';;
  2) names='s2 fcXG83vInDY c1 c4 s5 s6';;
esac
for name in $names; do
  run_limited "$OUT/combo_${name}.log" nice -n 10 "$PY" -B -m scripts.a1b_replay combo "$name"
done
