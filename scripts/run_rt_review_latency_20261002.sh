#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT=$ROOT/logs/rt_review_20261002/with_assets
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
for version in production static_lr; do
  export RT_STATIC_LR=0
  if [ "$version" = static_lr ]; then export RT_STATIC_LR=1; fi
  export RT_VERIFY_OUTPUT=$OUT/$version/audit
  mkdir -p "$OUT/$version"
  "$PYTHON" -c 'from pathlib import Path; a=int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))*1024; print(a); assert a >= 6*1024**3' > "$OUT/$version/resource_gate.txt"
  result=0
  nice -n 10 "$PYTHON" scripts/verify_rt_integration_live_20261002.py --source video --realtime --start-sec 2579 --end-sec 2650 --warmup-sec 1 --cpu-threads 1 --evaluation-nice 10 --cnn-device cpu --recognition-audit --runtime-audit --output "$OUT/$version" --port 8878 > "$OUT/$version/run.log" 2>&1 || result=$?
  echo "$result" > "$OUT/$version/exit.txt"
  if [ "$result" != 0 ]; then exit "$result"; fi
done
