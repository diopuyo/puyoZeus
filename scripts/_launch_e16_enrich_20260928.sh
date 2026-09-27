#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
mkdir -p logs/e16/records
export PYTHONPATH=.
/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -u -m scripts.enrich_e16_records_20260928 > logs/e16/records/enrich.log 2>&1 < /dev/null
