set -euo pipefail
shopt -s nullglob
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m pytest tests/test_e10*.py tests/test_e12*.py tests/test_e17*.py tests/test_e18*.py tests/test_e19*.py tests/test_e20*.py tests/test_e21*.py tests/test_e22*.py tests/test_e23*.py tests/test_e25*.py tests/test_e26*.py tests/test_exchange_event*.py -q > logs/e26/tests.log 2>&1 < /dev/null
