#!/usr/bin/env bash
# W25検証スクリプトの detach 起動ラッパー (MSYS quote escape回避のためファイル化)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
setsid -f bash -c '
PYTHONPATH=. ./venv/bin/python -m scripts._verify_w25_fix_2026-08-17 \
  > logs/_w25_verify_2026-08-17.log 2>&1
echo JOB_DONE >> logs/_w25_verify_2026-08-17.log
' < /dev/null
echo LAUNCHED
