#!/bin/bash
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
mkdir -p data/verify/relphase_winpanel_2026-07-30
nice -n 19 ./venv/bin/python -m scripts._diag_relphase_by_winpanel_2026-07-30 \
  > logs/relphase_winpanel_2026-07-30.log 2>&1
echo "[done] exit=$?" >> logs/relphase_winpanel_2026-07-30.log
