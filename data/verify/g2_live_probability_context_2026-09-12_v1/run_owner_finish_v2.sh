#!/usr/bin/env bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 2
probe_root=data/verify/g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v69_owner_finish
probe_log=logs/g2_owner_finish_v69.log
probe_exit=logs/g2_owner_finish_v69.exit
probe_result=data/verify/g2_live_probability_context_2026-09-12_v1/POSITIVE_CORE_PROBE_v69.json
if [ -e "$probe_root" ] || [ -e "$probe_log" ] || [ -e "$probe_exit" ] || [ -e "$probe_result" ]; then exit 2; fi
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 G2_TARGET_BASIS_ONLY=1
timeout 1200s ./venv/bin/python data/verify/g2_live_probability_context_2026-09-12_v1/probe_owner_finish_v2.py prefix_cpu_v69_owner_finish > "$probe_log" 2>&1 < /dev/null
probe_code=$?
printf '%s\n' "$probe_code" > "$probe_exit"
exit "$probe_code"
