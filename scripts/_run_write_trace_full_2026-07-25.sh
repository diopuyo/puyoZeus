#!/bin/bash
# write_trace フル走行(4動画・既知窓を明示指定、順次実行)
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
PY=./venv/bin/python
run() {
  echo "[launcher] $1 start=$2 dur=$3 $(date '+%H:%M:%S')"
  nice -n 15 $PY scripts/_diag_confirmed_write_trace_2026-07-25.py --video "$1" --start-sec "$2" --max-sec "$3"
}
run c62 862.0 93.0
run 30 225.0 90.0
run 35 3110.0 90.0
run 38 2585.0 90.0
echo "[launcher] ALL DONE $(date '+%H:%M:%S')"
