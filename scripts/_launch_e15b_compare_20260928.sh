#!/usr/bin/env bash
set -euo pipefail
cd <home>/.codex/worktrees/exev/puyo_analyzer
mkdir -p logs/e15b
export PYTHONPATH=.
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.compare_e15b_models_20260928 > logs/e15b/compare.log 2>&1 < /dev/null
