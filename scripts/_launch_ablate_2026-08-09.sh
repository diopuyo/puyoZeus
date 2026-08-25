#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
setsid -f bash ./scripts/_run_ablate_shards_2026-08-09.sh > logs/ablate_main_2026-08-09.log 2>&1 < /dev/null
echo "launched rc=$?"
