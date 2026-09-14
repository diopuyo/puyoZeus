#!/usr/bin/env bash
# 原resource guard/実wait/親finalizeを保持する固定区間。外側setsid -f -wで起動する。
set -euo pipefail
project=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
runner=data/verify/g2_second_prefix_runtime_2026-09-14_v38/target_entry.py
guard=data/verify/g2_bounded_publication_runtime_2026-09-09_v1/resource_guard.py
supervisor=data/verify/g2_m1_paced_runtime_2026-09-13_v12/supervise.py
output=data/verify/video38_second_prefix_candidate_v38
cd "$project"
[[ ! -e "$output" && ! -e "$output.log" && ! -e "$output.resources.jsonl" ]] || exit 91
set -o noclobber
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export PYTHONOPTIMIZE=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
(
  set +e
  ./venv/bin/python -u "$supervisor" "$runner" "$output" "$guard"
  final=$?
  echo ACTUAL_SUPERVISOR_EXIT=$final
  exit "$final"
) > "$output.log" 2>&1 < /dev/null
