#!/bin/bash
# 共通回帰と原票5記録の既定OFFバイト一致を実際に実行する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
OUT=logs/eval_set/falsedeath_fix/v2
available=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
test "$available" -ge 6291456
if [ "${1:-}" = style ]; then
  nice -n 10 "$PY" -B -m pytest -q -p no:cacheprovider \
    --basetemp "$OUT/style_final_tmp" \
    tests/test_exchange_event_tracker.py::test_new_functions_within_fifty_lines > "$OUT/test_style.log" 2>&1
  exit
fi
printf '{"memory_available_kib":%s,"nice":10,"blas_threads":1}\n' "$available" > "$OUT/resources_unit.json"
nice -n 10 "$PY" -B -m pytest -q -p no:cacheprovider \
  --basetemp "$OUT/unit_final_tmp" \
  tests/test_falsedeath_v2.py tests/test_falsedeath_d1.py tests/test_falsedeath_d2.py \
  tests/test_falsedeath_d3.py tests/test_falsedeath_d4.py \
  tests/test_e35_post_counter_bound.py tests/test_e35b_geometry.py tests/test_e35_integration.py \
  tests/test_post_counter_early_exit.py tests/test_exchange_event_tracker.py \
  tests/test_exchange_event_overlay.py tests/test_exchange_event_production.py > "$OUT/tests_final.log" 2>&1
