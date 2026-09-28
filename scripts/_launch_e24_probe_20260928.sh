set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
"$python_bin" -m scripts.inspect_e24_saved_20260928 > logs/e24/inspect.log 2>&1 < /dev/null
while [ ! -f logs/e24/q_7gc4TgFig_runtime.json ] || [ ! -f logs/e24/fcXG83vInDY_runtime.json ]; do
  sleep 5
done
"$python_bin" -m scripts.probe_e24_inputs_20260928 > logs/e24/probe.log 2>&1 < /dev/null
