#!/usr/bin/env bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 2
attach_root=data/verify/g2_live_probability_context_2026-09-12_v1
attach_log=logs/g2_live_probability_attach_v1.log
attach_exit=logs/g2_live_probability_attach_v1.exit
if [ -e "$attach_root/attach_actual_v1" ] || [ -e "$attach_log" ] || [ -e "$attach_exit" ]; then exit 2; fi
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
timeout 600s ./venv/bin/python "$attach_root/probe_attach.py" actual_v1 > "$attach_log" 2>&1 < /dev/null
attach_code=$?
printf '%s\n' "$attach_code" > "$attach_exit"
exit "$attach_code"
