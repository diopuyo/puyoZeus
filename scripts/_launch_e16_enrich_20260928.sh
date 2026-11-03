#!/usr/bin/env bash
set -euo pipefail
cd <home>/.codex/worktrees/exev/puyo_analyzer
mkdir -p logs/e16/records
export PYTHONPATH=.
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -u -m scripts.enrich_e16_records_20260928 > logs/e16/records/enrich.log 2>&1 < /dev/null
