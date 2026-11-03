#!/usr/bin/env bash
set -eu
cd <home>/.codex/worktrees/rt/puyo_analyzer
mkdir -p "$1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
output="$1"
shift
exec <home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.profile_live_b14 --output "$output" "$@" > "$output/run.log" 2>&1 < /dev/null
