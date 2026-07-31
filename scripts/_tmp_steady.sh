#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
echo "procs=$(pgrep -c -f scripts._collect_1t)"
ps -eo pcpu,args | grep -F scripts._collect_1t | grep -F python | grep -v grep | awk '{s+=$1} END {print "合計CPU%="s}'
free -g | head -2
