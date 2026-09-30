#!/bin/bash
# 走行中の python の常駐メモリ (KB) を 30 秒ごとに記録する (計測負荷は無視できる)
while true; do
  ps -eo pid,rss,etime,args | grep -E "run_multilanding_speed_gate|profile_multilanding" | grep -v grep | cut -c1-170 | sed "s/^/$(date +%H:%M:%S) /"
  sleep 30
done >> /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/multilanding_speed/rss.log
