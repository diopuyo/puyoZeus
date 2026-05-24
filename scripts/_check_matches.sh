#!/usr/bin/env bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
for v in 90 91 92 93; do
    fpath="data/verify/match_boundaries_v5/video_${v}/matches.tsv"
    if [ -f "$fpath" ]; then
        echo "=== v${v} ==="
        head -5 "$fpath"
    fi
done
