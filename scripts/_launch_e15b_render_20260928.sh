#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
mkdir -p logs/review_zenchi_g41_43_e15
export PYTHONPATH=.
/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.render_e15_review_20260928 > logs/review_zenchi_g41_43_e15/launcher.log 2>&1 < /dev/null
