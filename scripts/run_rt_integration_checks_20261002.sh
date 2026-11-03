#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
# 他worktreeのPythonは実行にだけ使い、検証結果はrt内へ保存する。
"$PYTHON" -c 'from pathlib import Path; a=int(next(s.split()[1] for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:")))*1024; print("available_bytes", a); assert a >= 6*1024**3'
exec nice -n 10 "$PYTHON" "$@"
