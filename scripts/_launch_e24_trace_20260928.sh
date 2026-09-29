set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
for source in review q_7gc4TgFig fcXG83vInDY; do
  "$python_bin" -m scripts.trace_e24_runtime_20260928 "$source" > "logs/e24/${source}_trace.log" 2>&1 < /dev/null &
done
wait
