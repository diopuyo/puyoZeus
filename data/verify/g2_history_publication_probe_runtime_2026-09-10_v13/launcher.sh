#!/usr/bin/env bash
# 実GPU開始は親のみ。このfile自体は製造CPUから呼ばない。
set -euo pipefail
project=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
runner=data/verify/g2_history_publication_probe_runtime_2026-09-10_v13/live_cli.py
guard=data/verify/g2_bounded_publication_runtime_2026-09-09_v1/resource_guard.py
output=${1:?new exclusive output root required}
cd "$project"
[[ ! -e "$output" && ! -e "$output.log" && ! -e "$output.resources.jsonl" ]] || exit 1
set -o noclobber
setsid -f bash -c '
  set +e
  export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
  export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
  ./venv/bin/python -u "$1" --mode live --output-root "$2" &
  child=$!
  echo ACTUAL_CHILD_PID=$child
  ./venv/bin/python "$3" "$child" "$1" "$2.resources.jsonl" &
  guard=$!
  wait "$child"; code=$?
  echo ACTUAL_CHILD_EXIT=$code
  wait "$guard"; resource=$?
  echo ACTUAL_RESOURCE_GUARD_EXIT=$resource
  ./venv/bin/python -u "$1" --mode finalize --output-root "$2" --child-exit "$code" --resource-exit "$resource"
  final=$?
  echo ACTUAL_FINALIZE_EXIT=$final
  exit "$final"
' probe "$runner" "$output" "$guard" > "$output.log" 2>&1 < /dev/null
