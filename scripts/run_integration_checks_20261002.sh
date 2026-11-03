#!/usr/bin/env bash
set -eu
cd /mnt/d/puyo_analyzer/wt_integ
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
# 他worktreeの環境は実行ファイルとして読むだけで、キャッシュを書かない。
"$PYTHON" -c 'from pathlib import Path; a=int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))*1024; print("available_bytes", a); assert a >= 6*1024**3'
exec nice -n "${INTEG_NICE:-10}" "$PYTHON" "$@"
