#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
LOG=data/phase_e_count_match_v20p.log
> "$LOG"

for v in 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40; do
    echo "=== v${v} ===" | tee -a "$LOG"
    ./venv/bin/python scripts/count_match_v4.py \
        --video "data/frames/video_${v}.mp4" \
        --out-root data/verify/match_boundaries_v5 \
        --interval 1 --confirm 3 \
        --min-duration 20 --max-duration 220 2>&1 | tail -3 | tee -a "$LOG"
done
echo "=== complete ===" | tee -a "$LOG"
