#!/usr/bin/env bash
set -euo pipefail
cd <home>/.codex/worktrees/exev/puyo_analyzer
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.run_e21_death_candidate_20260928 --source review --off > logs/e21/off.log 2>&1 < /dev/null
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.audit_e21_corpus_20260928 > logs/e21/corpus.log 2>&1 < /dev/null
