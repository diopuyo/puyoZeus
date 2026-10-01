#!/usr/bin/env bash
# 切替平滑の単体テストと関連の既存テスト (1 プロセス・nice 10)
set -eu
cd /mnt/d/puyo_analyzer/wt_switch
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 nice -n 10 $PY -B -m pytest -q -p no:cacheprovider "$@"
