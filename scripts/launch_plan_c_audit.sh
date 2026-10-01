#!/usr/bin/env bash
# 優先ジョブと競合しない1プロセス・nice 10の保存データ監査。
set -eu
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out=logs/prefire_prediction/plan_c
mkdir -p "$out"
exec 9>"$out/audit.lock"
flock -n 9 || exit 1
set +e
nice -n 10 /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.plan_c_input_audit > "$out/input_audit.log" 2>&1
code=$?
printf '%s\n' "$code" > "$out/audit.exit"
if [ "$code" -eq 0 ]; then
    nice -n 10 /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.plan_c_benchmark > "$out/benchmark.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "$out/benchmark.exit"
fi
exit "$code"
