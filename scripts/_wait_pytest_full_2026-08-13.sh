#!/bin/bash
# W7 タスク検証用: 検出済み detached pytest プロセスの終了を待って結果を出力する
# 使い捨てスクリプト (MSYS パイプ特殊文字回避のためファイル化、feedback_msys_pipe_escape.md)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PID=$(pgrep -f "pytest tests/ -q" | tail -1)
if [ -z "$PID" ]; then
  echo "NO_PYTEST_PROCESS_FOUND"
  exit 1
fi
echo "WATCHING_PID=$PID"
tail --pid="$PID" -f /dev/null
echo "PYTEST_FULL_DONE"
tail -n 100 logs/_pytest_full_w7_2026-08-13.log
