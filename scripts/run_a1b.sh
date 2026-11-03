#!/bin/bash
# 二構成を各一ワーカーで逐次再生し、同じ資源制限を守る。
set -eu
cd /mnt/d/puyo_analyzer/wt_a1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
OUT=$PWD/logs/a1b
mkdir -p "$OUT"
source scripts/a1_resource_limits.sh
if [ "$1" = report ]; then
  run_limited "$OUT/report.log" nice -n 10 "$PY" -B -m scripts.a1b_report
  exit
fi
if [ "$1" = test ]; then
  run_limited "$OUT/tests.log" nice -n 10 "$PY" -B -m pytest tests/test_completion_score_max_chain.py tests/test_completion_score_coherent.py tests/test_completion_score_cli.py tests/test_a2_safety.py tests/test_e30_prefire_candidates.py tests/test_e31_prefire_snapshot.py tests/test_e32_hidden_row_belief.py tests/test_hidden_row_probability.py -q
  exit
fi
mode=$1
mkdir -p "$OUT/$mode/replay"
if [ ! -e "$OUT/$mode/replay/prod" ]; then ln -s "$PWD/logs/a1_completion/replay/prod" "$OUT/$mode/replay/prod"; fi
for name in review q_7gc4TgFig fcXG83vInDY mia8KCjr52g zenchi c1 c2 c3 c4 c5 c6 s0 s1 s2 s3 s4 s5 s6; do
  run_limited "$OUT/${mode}_${name}.log" nice -n 10 "$PY" -B -m scripts.a1b_replay "$mode" "$name"
done
run_limited "$OUT/${mode}_score.log" nice -n 10 "$PY" -B -m scripts.a1b_score "$mode"
