#!/usr/bin/env bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 91
root=data/verify/g2_empty_tail_reset_integration_2026-09-11_v1
test ! -e "$root/prefix_cpu_v13" && test ! -e "$root/entry_v13.log" || exit 92
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
./venv/bin/python "$root/run_entry_v2.py" prefix_cpu_v13 > "$root/entry_v13.log" 2>&1 < /dev/null
result=$?
printf '\nACTUAL_CHILD_EXIT=%s\n' "$result" >> "$root/entry_v13.log"
exit "$result"
