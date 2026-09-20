#!/usr/bin/env bash
set -euo pipefail
project_root=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
runner="$project_root/data/verify/g2_provisional_context_capture_2026-09-09_v1/live_cli.py"
output="${1:?exclusive new output root}"
script_sha="${2:?split runtime adapter SHA}"
test_sha="${3:?split runtime test SHA}"
launcher_sha="${4:?split runtime launcher SHA}"
cd "$project_root"
[[ ! -e "$output" && ! -e "$output.log" ]] || exit 1
set -o noclobber
setsid -f bash -c '
  set +e
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
  ./venv/bin/python -u "$1" --output-root "$2" --allow-native-runtime-mismatch --script-sha256 "$3" --test-sha256 "$4" --launcher-sha256 "$5" &
  child=$!; printf "ACTUAL_CHILD_PID=%s\n" "$child"
  wait "$child"; status=$?; printf "ACTUAL_CHILD_EXIT=%s\n" "$status"
  ./venv/bin/python -u "$1" --output-root "$2" --finalize-child-exit "$status"
  finalized=$?; printf "ACTUAL_FINALIZE_EXIT=%s\n" "$finalized"
  [[ "$status" -eq 0 ]] || exit "$status"
  exit "$finalized"
' provisional-context "$runner" "$output" "$script_sha" "$test_sha" "$launcher_sha" >"$output.log" 2>&1 </dev/null
