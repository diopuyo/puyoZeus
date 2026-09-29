set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for source in review q_7gc4TgFig fcXG83vInDY mia8KCjr52g zenchi; do
  /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.run_e32 --source "$source" --control > "logs/e32/off_${source}_runner.log" 2>&1 < /dev/null
done
