#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
echo "DATE: $(date)"
echo "RUNNING: $(ps aux | grep zap_1t | grep -v grep | wc -l)"
echo "DONE_MARKER: $(grep -c 'ALL DONE' logs/zap_reel/_driver.log 2>/dev/null)"
ls -la data/indicators_v2/overlay/zap/raw/ 2>/dev/null
