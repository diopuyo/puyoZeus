set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
$PY -m scripts.verify_e26_records > logs/e26/record_verification.log 2>&1
$PY -m scripts.run_e26_completion > logs/e26/completion.log 2>&1
