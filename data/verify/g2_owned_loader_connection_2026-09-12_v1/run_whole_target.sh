#!/usr/bin/env bash
# 原resource guard/実wait/親finalizeを保持する固定区間。外側setsid -f -wで起動する。
set -euo pipefail
project=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
runner=data/verify/g2_owned_loader_connection_2026-09-12_v1/target_entry.py
guard=data/verify/g2_bounded_publication_runtime_2026-09-09_v1/resource_guard.py
output=data/verify/video38_history_publication_probe_whole_start_v3
cd "$project"
[[ ! -e "$output" && ! -e "$output.log" && ! -e "$output.resources.jsonl" ]] || exit 91
set -o noclobber
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export PYTHONOPTIMIZE=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
(
  set +e
  ./venv/bin/python -u "$runner" --mode observe --output-root "$output" &
  child=$!
  echo ACTUAL_CHILD_PID=$child
  ./venv/bin/python "$guard" "$child" "$runner" "$output.resources.jsonl" &
  guard_pid=$!
  wait "$child"; code=$?
  echo ACTUAL_CHILD_EXIT=$code
  wait "$guard_pid"; resource=$?
  echo ACTUAL_RESOURCE_GUARD_EXIT=$resource
  ./venv/bin/python -u "$runner" --mode finalize --output-root "$output" --child-exit "$code" --resource-exit "$resource"
  final=$?
  echo ACTUAL_FINALIZE_EXIT=$final
  exit "$final"
) > "$output.log" 2>&1 < /dev/null
