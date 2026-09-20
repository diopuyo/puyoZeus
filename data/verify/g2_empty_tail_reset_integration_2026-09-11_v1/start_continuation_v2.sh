#!/usr/bin/env bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 2
root=data/verify/g2_empty_tail_reset_integration_2026-09-11_v1
test ! -e "$root/prefix_cpu_v2" || exit 3
test ! -e "$root/continuation_v2.log" || exit 3
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
./venv/bin/python "$root/run_continuation.py" prefix_cpu_v2 > "$root/continuation_v2.log" 2>&1
result=$?
printf 'ACTUAL_CHILD_EXIT=%s\n' "$result" >> "$root/continuation_v2.log"
exit "$result"
