#!/bin/bash
# WSL venv で python を nice19・単一プロセスで実行する薄いラッパ (使い方: py.sh -m モジュール 引数...)
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec nice -n 19 /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -B "$@"
