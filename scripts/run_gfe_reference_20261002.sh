#!/bin/bash
set -eu
cd /mnt/d/puyo_analyzer/wt_phase
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec nice -n 19 <home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.gfe_phase_reference_20261002 > logs/gfe_phase_reference_20261002/console.log 2>&1 < /dev/null
