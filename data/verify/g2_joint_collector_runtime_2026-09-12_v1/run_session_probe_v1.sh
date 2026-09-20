#!/usr/bin/env bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 2
probe_log=logs/g2_joint_session_probe_v1.log
probe_exit=logs/g2_joint_session_probe_v1.exit
if [ -e "$probe_log" ] || [ -e "$probe_exit" ]; then exit 2; fi
PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 1200s ./venv/bin/python data/verify/g2_joint_collector_runtime_2026-09-12_v1/probe_session_constructor.py prefix_cpu_v62_session_probe > "$probe_log" 2>&1 < /dev/null
probe_code=$?
printf '%s\n' "$probe_code" > "$probe_exit"
exit "$probe_code"
