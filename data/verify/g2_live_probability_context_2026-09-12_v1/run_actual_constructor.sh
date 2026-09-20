#!/usr/bin/env bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 90
target=data/verify/g2_live_probability_context_2026-09-12_v1/constructor_observation_v1
log=logs/g2_actual_constructor_v1.log
test ! -e "$target" && test ! -e "$log" && test ! -e logs/g2_actual_constructor_v1.exit || exit 91
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
timeout 900 ./venv/bin/python -u data/verify/g2_live_probability_context_2026-09-12_v1/probe_actual_constructor.py constructor_observation_v1 > "$log" 2>&1 < /dev/null
code=$?
printf '%s\n' "$code" > logs/g2_actual_constructor_v1.exit
exit "$code"
