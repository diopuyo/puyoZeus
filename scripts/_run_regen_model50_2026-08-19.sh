#!/bin/bash
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
exec >> logs/regen_model50_2026-08-19.log 2>&1
echo "[run_regen_model50] 開始 $(date)"
./venv/bin/python -u scripts/_regen_model50_2026-08-19.py
echo "[run_regen_model50] 終了 $(date) rc=$?"
