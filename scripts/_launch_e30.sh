set -euo pipefail
cd <home>/.codex/worktrees/exev/puyo_analyzer
mkdir -p logs/e30
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.run_e30 > logs/e30/worker.log 2>&1 < /dev/null
