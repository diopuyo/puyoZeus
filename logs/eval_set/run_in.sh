#!/bin/bash
# 評価セットの Python 実行口 (cwd = exev 参照の実行ディレクトリ、コードは wt_evalset)。
# 使い方: bash run_in.sh <python 引数...>
cd /mnt/d/puyo_analyzer/evalset_run
export PYTHONPATH=/mnt/d/puyo_analyzer/wt_evalset PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -B "$@"
