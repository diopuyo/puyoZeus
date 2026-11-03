set -euo pipefail
cd <home>/.codex/worktrees/exev/puyo_analyzer
mkdir -p logs/e28
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.render_e28_review > logs/e28/render_worker.log 2>&1 < /dev/null
