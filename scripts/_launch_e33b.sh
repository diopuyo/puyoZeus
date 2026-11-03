#!/usr/bin/env bash
set -eu
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.run_e33b > logs/e33b/runner.log 2>&1 < /dev/null
