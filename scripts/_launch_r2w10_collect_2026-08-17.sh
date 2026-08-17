#!/bin/bash
# 認識強化統一測定 (2026-08-17) の B/C/D 収集をバックグラウンド起動する。
# MSYSパイプ・引用符の事故回避のため、複雑なコマンドをファイル化する
# (feedback_msys_pipe_escape.md)。
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
mkdir -p logs/yardstick_v2_collect_r2w10_2026-08-17
setsid -f bash -c 'cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m scripts._collect_yardstick_v2_r2w10_2026-08-17 --config all > logs/yardstick_v2_collect_r2w10_2026-08-17/_driver.log 2>&1 < /dev/null'
echo "launched"
