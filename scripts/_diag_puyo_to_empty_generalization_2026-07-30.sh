#!/bin/bash
set -x
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
OUTDIR="data/verify/placement_confirm_frames_generalization_2026-07-30"
mkdir -p "$OUTDIR"
export PYTHONPATH=.

run_one() {
  local video=$1 start=$2 dur=$3 flag=$4 suffix=$5
  echo "==== START video=$video flag=$flag $(date) ===="
  if [ "$flag" = "on" ]; then
    nice -n 19 ./venv/bin/python scripts/_diag_placement_confirm_frames_2026-07-25.py \
      --video "$video" --start-sec "$start" --max-sec "$dur" \
      --enable-puyo-to-empty-hsv-guard \
      --output-suffix "_${suffix}" --output-dir "$OUTDIR" 2>&1
  else
    nice -n 19 ./venv/bin/python scripts/_diag_placement_confirm_frames_2026-07-25.py \
      --video "$video" --start-sec "$start" --max-sec "$dur" \
      --output-suffix "_${suffix}" --output-dir "$OUTDIR" 2>&1
  fi
  echo "==== END video=$video flag=$flag $(date) ===="
}

run_one c26 238.0 72.0 off c26_off
run_one c26 238.0 72.0 on  c26_on
run_one c58 438.0 68.0 off c58_off
run_one c58 438.0 68.0 on  c58_on
run_one c69 278.0 68.0 off c69_off
run_one c69 278.0 68.0 on  c69_on
run_one c56 282.0 80.0 off c56_off
run_one c56 282.0 80.0 on  c56_on

echo "ALL DONE $(date)"
