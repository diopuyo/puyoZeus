#!/bin/bash
# 実再生を含まない構成選択の単体テスト。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
OUT=logs/eval_set/falsedeath_fix/v2b
name="${1:-tests}"
test ! -e "$OUT/$name.log"
available=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
test "$available" -ge 6291456
if [ "$name" = tables ] || [ "${2:-tests}" = tables ]; then
  nice -n 10 "$PY" -B -m scripts._diag_falsedeath_v2b_tables > "$OUT/$name.log" 2>&1
  exit
fi
nice -n 10 "$PY" -B -m pytest -q -p no:cacheprovider --basetemp "$OUT/${name}_tmp" \
  tests/test_falsedeath_modes.py > "$OUT/$name.log" 2>&1
