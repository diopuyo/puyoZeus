set -euo pipefail
cd <home>/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
for source in review zenchi; do
  $PY -m scripts.run_e26_completion --source "$source" >> logs/e26/scene_worker.log 2>&1 < /dev/null
done
