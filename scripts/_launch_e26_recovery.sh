set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
for source in q_7gc4TgFig fcXG83vInDY mia8KCjr52g zenchi review; do
  $PY -m scripts.run_e26_ablation --variant recovery --source "$source" >> logs/e26/recovery_worker.log 2>&1 < /dev/null
done
