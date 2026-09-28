set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m pytest tests/test_e27* tests/test_e30* tests/test_e31* tests/test_e32* tests/test_exchange_event* tests/test_exchange_virtual_board.py tests/test_hidden_row_probability.py tests/test_hidden_row_tracker.py tests/test_probabilistic_board.py tests/test_e11_review_data_panel.py -q -x > logs/e32/tests.log 2>&1 < /dev/null
