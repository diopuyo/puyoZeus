#!/bin/bash
# 空き枠で1記録だけ再生する。Python側の記録別ロックで二重実行を防ぐ。
set -eu
cd /mnt/d/puyo_analyzer/wt_margin
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 10 /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -B -m scripts.measure_margin_clock --variant "$1" --source "$2" > "logs/margin_clock/${1}_${2}_parallel.log" 2>&1
