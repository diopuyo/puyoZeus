#!/bin/bash
# 拡張13本のregen (4並列・低優先度)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
nice -n 10 xargs -P 4 -d '\n' -I{} bash -c '{}' < scripts/_jobs_expand13_2026-08-03.txt
echo "[expand13] ALL DONE"
