#!/bin/bash
# 設置取りこぼし診断 (2026-07-24) 起動スクリプト。setsid -f detach 用。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
PYTHONPATH=. ./venv/bin/python -u scripts/_diag_place_coldstart_dropout_2026-07-24.py \
  > logs/_diag_dropout_2026-07-24.log 2>&1
