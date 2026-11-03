set -euo pipefail
cd <home>/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m pytest tests/test_e31_prefire_snapshot.py tests/test_e11_review_data_panel.py tests/test_exchange_event_record.py tests/test_exchange_event_tracker.py -q > logs/e31/final_tests.log 2>&1 < /dev/null
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.verify_e31 --code-only > logs/e31/function_check.log 2>&1 < /dev/null
