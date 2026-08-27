#!/bin/bash
# 汎用: 指定ログファイルに完了マーカーが出るまでポーリング待機する。
# 使い方: bash scripts/_wait_log_2026-08-17.sh <logfile> <max_iters>
LOGFILE="$1"
MAX_ITERS="${2:-60}"
i=0
while [ "$i" -lt "$MAX_ITERS" ]; do
  if grep -q -- '-> ' "$LOGFILE" 2>/dev/null; then
    break
  fi
  sleep 5
  i=$((i+1))
done
tail -n 30 "$LOGFILE"
