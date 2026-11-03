#!/bin/bash
# 各ワーカーは逐次実行し、もう一レーンと資源を共有する。
set -eu
cd /mnt/d/puyo_analyzer/wt_a1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
OUT=$PWD/logs/a1_completion
mkdir -p "$OUT"
source scripts/a1_resource_limits.sh
if [ "${1:-}" = progress ]; then
  run_limited "$OUT/progress.log" nice -n 10 "$PY" -B -m scripts.a1_completion_progress
  exit
fi
if [ "${1:-}" = preflight ]; then
  run_limited "$OUT/preflight.log" nice -n 10 "$PY" -B -m scripts.a1_completion_preflight
  exit
fi
if [ "${1:-}" = finish ] || [ "${1:-}" = tables ]; then
  modules='errors report'
  if [ "$1" = finish ]; then modules="score $modules"; fi
  for module in $modules; do
    run_limited "$OUT/${module}.log" nice -n 10 "$PY" -B -m "scripts.a1_completion_${module}"
  done
  exit
fi
if [ "${1:-}" = cli_test ]; then
  run_limited "$OUT/cli_tests.log" nice -n 10 "$PY" -B -m pytest tests/test_completion_score_cli.py tests/test_exchange_event_production.py tests/test_exchange_event_static_gfe_cli.py -q
  exit
fi
if [ "${1:-}" = test ]; then
  run_limited "$OUT/tests.log" nice -n 10 "$PY" -B -m pytest tests/test_completion_score_coherent.py tests/test_e30_prefire_candidates.py tests/test_e31_prefire_snapshot.py tests/test_e32_hidden_row_belief.py tests/test_hidden_row_probability.py -q
  exit
fi
for mode in ${A1_MODES:-original prod a1}; do
  names='review q_7gc4TgFig fcXG83vInDY mia8KCjr52g zenchi'
  if [ "$mode" != original ]; then names="$names c1 c2 c3 c4 c5 c6 s0 s1 s2 s3 s4 s5 s6"; fi
  names=${A1_NAMES:-$names}
  for name in $names; do
    if [ -f "$OUT/replay/$mode/$name/status.json" ]; then continue; fi
    run_limited "$OUT/${A1_LOG_PREFIX:-}${mode}_${name}.log" nice -n 10 "$PY" -B -m scripts.a1_completion_replay "$mode" "$name"
  done
done
