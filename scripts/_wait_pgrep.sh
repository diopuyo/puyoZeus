#!/bin/bash
# 使い捨て待機スクリプト: 指定パターンのプロセスが終わるまで待つ (MSYSパイプ回避のためファイル化)
PATTERN="$1"
MAXCNT="${2:-80}"
cnt=0
while pgrep -f "$PATTERN" >/dev/null && [ "$cnt" -lt "$MAXCNT" ]; do
    sleep 5
    cnt=$((cnt+1))
done
echo "done_waiting cnt=$cnt pattern=$PATTERN"
