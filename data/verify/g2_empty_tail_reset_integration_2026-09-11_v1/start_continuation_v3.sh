#!/usr/bin/env bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 91
root=data/verify/g2_empty_tail_reset_integration_2026-09-11_v1
test ! -e "$root/prefix_cpu_v3" && test ! -e "$root/continuation_v3.log" || exit 92
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
./venv/bin/python "$root/run_continuation.py" prefix_cpu_v3 > "$root/continuation_v3.log" 2>&1 < /dev/null
result=$?
printf '\nACTUAL_CHILD_EXIT=%s\n' "$result" >> "$root/continuation_v3.log"
exit "$result"
