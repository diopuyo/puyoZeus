#!/usr/bin/env bash
# PC1で逐次実行し、計測どうしの競合を避ける。既存出力は上書きしない。
set -euo pipefail
cd /mnt/d/puyo_analyzer/wt_b3
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 B3_HGB_THREADS=1
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
OUT=/mnt/d/puyo_analyzer/b3_data/time_gate
"$PY" -m scripts.b3_time_gate --mode realtime --start 300 --duration 1800 --out "$OUT/realtime_30min" > "$OUT/realtime_30min.log" 2>&1
"$PY" -m scripts.b3_time_gate --mode unpaced --video b3_zenchi_s1 --out "$OUT/full_s1" > "$OUT/full_s1.log" 2>&1
"$PY" -m scripts.b3_time_gate --mode unpaced --video b3_zenchi_s2 --out "$OUT/full_s2" > "$OUT/full_s2.log" 2>&1
touch "$OUT/all_complete"
