#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
for v in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19; do
    p5="data/verify/match_boundaries_v5/video_${v}/matches.tsv"
    p4="data/verify/match_boundaries_v4/video_${v}/matches.tsv"
    if [ -f "$p5" ]; then
        echo "v${v} v5: $(awk 'NR==2 {print $0}' "$p5")"
    elif [ -f "$p4" ]; then
        echo "v${v} v4: $(awk 'NR==2 {print $0}' "$p4")"
    else
        echo "v${v} NONE"
    fi
done
