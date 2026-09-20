#!/usr/bin/env bash
set -euo pipefail
project_root=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
runner="$project_root/data/verify/g2_bounded_publication_runtime_2026-09-09_v1/live_cli.py"
output="${1:?exclusive new output root}"
cd "$project_root"
[[ ! -e "$output" && ! -e "$output.log" && ! -e "$output.resources.jsonl" ]] || exit 1
set -o noclobber
setsid -f bash -c '
  set +e
  export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
  ./venv/bin/python -u "$1" --output-root "$2" --allow-native-runtime-mismatch --script-sha256 f24bd02f9bbf36c5a32028ede54ce2cc736e07657b1210ec4ee65279863a60d1 --test-sha256 3304093cb6c833e6ccee63118d590eb5cb2cf54e2b1e8960f8028c286dcab22a --launcher-sha256 5916fad296d2a2c4dc6b938f0d492ad71549a1504987cfc2092352cbaab90fea &
  child=$!; printf "ACTUAL_CHILD_PID=%s\n" "$child"
  ./venv/bin/python "${1%/*}/resource_guard.py" "$child" "$1" "$2.resources.jsonl" &
  guard=$!
  wait "$child"; status=$?; printf "ACTUAL_CHILD_EXIT=%s\n" "$status"
  wait "$guard"; guarded=$?; printf "ACTUAL_RESOURCE_GUARD_EXIT=%s\n" "$guarded"
  ./venv/bin/python -u "$1" --output-root "$2" --finalize-child-exit "$status"
  finalized=$?; printf "ACTUAL_FINALIZE_EXIT=%s\n" "$finalized"
  [[ "$status" -eq 0 ]] || exit "$status"
  exit "$finalized"
' metadata-publication "$runner" "$output" >"$output.log" 2>&1 </dev/null
