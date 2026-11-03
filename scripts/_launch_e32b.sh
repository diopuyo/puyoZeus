#!/bin/bash
set -euo pipefail
cd <home>/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
$PY -m scripts.audit_e32b --cutoffs 2849.05 3100.8166666666666 3110.05
$PY -m scripts.audit_e32b --run > logs/e32b/runner.log 2>&1 < /dev/null
