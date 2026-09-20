#!/usr/bin/env bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 2
joint_root=data/verify/g2_joint_collector_runtime_2026-09-12_v1
joint_output=data/verify/g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v64_joint_saved
joint_log=logs/g2_joint_saved_v64.log
joint_exit=logs/g2_joint_saved_v64.exit
if [ -e "$joint_log" ] || [ -e "$joint_exit" ] || [ -e "$joint_output" ]; then exit 2; fi
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 G2_TARGET_BASIS_ONLY=1
timeout 7200s ./venv/bin/python "$joint_root/run_joint.py" prefix_cpu_v64_joint_saved > "$joint_log" 2>&1 < /dev/null
joint_code=$?
if [ "$joint_code" -eq 0 ]; then
  ./venv/bin/python "$joint_root/verify_joint_saved.py" "$joint_output" >> "$joint_log" 2>&1 < /dev/null
  joint_code=$?
fi
printf '%s\n' "$joint_code" > "$joint_exit"
exit "$joint_code"
