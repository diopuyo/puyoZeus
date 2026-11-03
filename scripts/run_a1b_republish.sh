#!/bin/bash
# 三ワーカーに記録を固定分割し、既存の資源制限と記録ロックを維持する。
set -eu
cd /mnt/d/puyo_analyzer/wt_a1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
OUT=$PWD/logs/a1b
source scripts/a1_resource_limits.sh
if [ "$1" = test ]; then
  run_limited "$OUT/republish_test.log" nice -n 10 "$PY" -B -m pytest tests/test_completion_score_max_chain.py::test_missing_public_prediction_republishes_same_candidate -q
  exit
fi
case "$1" in
  0) names='review q_7gc4TgFig mia8KCjr52g c2 c5 s0 s3';;
  1) names='zenchi c3 c6 s1 s4';;
  2) names='fcXG83vInDY c1 c4 s2 s5 s6';;
  finish)
    for mode in single combo; do
      run_limited "$OUT/${mode}_score.log" nice -n 10 "$PY" -B -m scripts.a1b_score "$mode"
    done
    run_limited "$OUT/a2_reference.log" nice -n 10 "$PY" -B -m scripts.a1b_a2_reference
    run_limited "$OUT/report.log" nice -n 10 "$PY" -B -m scripts.a1b_report
    exit;;
esac
for mode in single combo; do
  for name in $names; do
    run_limited "$OUT/${mode}_${name}.log" nice -n 10 "$PY" -B -m scripts.a1b_replay "$mode" "$name"
  done
done
