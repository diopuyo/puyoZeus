#!/bin/bash
set -eu
cd /mnt/d/puyo_analyzer/wt_phase
export PYTHONDONTWRITEBYTECODE=1
exec nice -n 19 <home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -B -m scripts.gfe_phase_stage2b_monitor_20261002 > logs/gfe_phase_stage2b_20261002/monitor.log 2>&1 < /dev/null
