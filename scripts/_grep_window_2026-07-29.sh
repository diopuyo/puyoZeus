#!/bin/bash
# ログから指定秒範囲の settled/SETTLED 行だけ抜き出す(MSYSパイプ回避のためファイル化)。
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
LOG="$1"
LO="$2"
HI="$3"
grep -E 'settled|SETTLED' "$LOG" | grep -v RESET | while IFS= read -r line; do
  t=$(echo "$line" | sed -n 's/.*t=\([0-9.]*\)s.*/\1/p')
  ok=$(awk -v t="$t" -v lo="$LO" -v hi="$HI" 'BEGIN{print (t>=lo && t<=hi) ? 1 : 0}')
  if [ "$ok" = "1" ]; then
    echo "$line"
  fi
done
