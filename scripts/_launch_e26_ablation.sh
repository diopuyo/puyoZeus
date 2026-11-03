set -euo pipefail
cd <home>/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m pytest tests/test_e25_completion_recovery.py tests/test_e25_landing_safety.py tests/test_e25_pending_ledger.py -q > logs/e26/split_tests.log 2>&1
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.run_e26_ablation > logs/e26/ablation.log 2>&1 < /dev/null
