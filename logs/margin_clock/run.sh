#!/bin/bash
# このworktreeだけへ出力する。最大2ワーカー、nice 10、数値ライブラリ1スレッド。
set -eu
cd /mnt/d/puyo_analyzer/wt_margin
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
variant=${1:-legacy_off}
run() { nice -n 10 "$PY" -B -m scripts.measure_margin_clock --variant "$variant" --source "$1" > "logs/margin_clock/${variant}_${1}.log" 2>&1; }
( run q_7gc4TgFig; run fcXG83vInDY; run mia8KCjr52g ) &
pid1=$!
( run review; run zenchi ) &
pid2=$!
wait "$pid1"
wait "$pid2"
nice -n 10 "$PY" -B -m scripts.measure_margin_clock --variant "$variant" > "logs/margin_clock/${variant}_report.log" 2>&1
