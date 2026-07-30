#!/bin/bash
set -x
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
export PYTHONPATH=.
mkdir -p logs

run_one() {
  local video=$1 start=$2 dur=$3 guard=$4
  nice -n 19 ./venv/bin/python scripts/_diag_puyo_to_empty_corruption_check_2026-07-30.py \
    --video "$video" --start-sec "$start" --max-sec "$dur" --guard "$guard"
}

run_one c34 470.0 46.0 off
run_one c34 470.0 46.0 on
run_one c26 238.0 72.0 off
run_one c26 238.0 72.0 on

echo "CORRUPTION_CHECK_ALL_DONE $(date)"
