#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
LOG=data/phase_e_winners_v20p.log
> "$LOG"

# v22 (1 試合のみ), v39 (0 試合) は skip
for v in 20 21 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 40; do
    echo "=== v${v} ===" | tee -a "$LOG"
    ./venv/bin/python scripts/detect_match_winners.py \
        --video "data/frames/video_${v}.mp4" \
        --matches-tsv "data/verify/match_boundaries_v5/video_${v}/matches.tsv" \
        --out "data/verify/match_winners_v${v}.tsv" 2>&1 | tail -3 | tee -a "$LOG"
done
echo "=== complete ===" | tee -a "$LOG"
