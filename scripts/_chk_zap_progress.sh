#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
for f in logs/zap_reel/*_render.log; do
  echo "== $f =="
  tail -3 "$f" 2>/dev/null
done
echo "---driver---"
tail -30 logs/zap_reel/_driver.log
