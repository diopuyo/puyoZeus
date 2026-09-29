#!/bin/bash
set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
$PY -m scripts.audit_e32b --cutoffs 2849.05 3100.8166666666666 3110.05
$PY -m scripts.audit_e32b --run > logs/e32b/runner.log 2>&1 < /dev/null
