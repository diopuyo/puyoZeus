set -euo pipefail
cd <home>/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
$PY -m scripts.run_e27 --source review > logs/e27/scene.log 2>&1
$PY -m scripts.run_e27 --source review --control > logs/e27/control.log 2>&1
