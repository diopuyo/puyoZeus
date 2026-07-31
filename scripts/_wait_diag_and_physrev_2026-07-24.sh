#!/bin/bash
# 修正C 検証待ち: diag script + physics_review before/after の完了を最大5分待つ。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
n=0
while { [ -d /proc/44743 ] || [ -d /proc/52303 ] || [ -d /proc/54758 ]; } && [ "$n" -lt 28 ]; do
  sleep 10
  n=$((n + 1))
done
echo "waited iterations: $n"
ps -o pid,etimes,pcpu -p 44743,52303,54758 2>&1
echo "---diag files---"
ls data/verify/diag_false_event_source_2026-07-24/ 2>&1
echo "---physrev before log tail---"
tail -15 logs/_physrev_C_before_2026-07-24.log 2>&1
echo "---physrev after log tail---"
tail -15 logs/_physrev_C_after_2026-07-24.log 2>&1
