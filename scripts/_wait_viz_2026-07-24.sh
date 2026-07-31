#!/bin/bash
# 修正C viz 完了待ち: 最大5分刻みで確認。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
n=0
while pgrep -f _viz_chain_debounce_before_after > /dev/null && [ "$n" -lt 28 ]; do
  sleep 10
  n=$((n + 1))
done
echo "waited iterations: $n"
pgrep -a -f _viz_chain_debounce_before_after 2>&1
echo "---log tail---"
tail -30 logs/_viz_chain_debounce_2026-07-24.log 2>&1
echo "---output dir---"
ls data/verify/chain_debounce_before_after_2026-07-24/ 2>&1
