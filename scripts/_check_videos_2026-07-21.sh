#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
for v in c10 c20 c24 c30 c33 c37 c42 c43 c44 c54 c55 c56 c57 c58 c62 c65 c68 c71 c76 c79 c81 c82 c84; do
    f="data/frames/video_${v}.mp4"
    if [ -f "$f" ]; then
        echo "OK $f"
    else
        echo "MISSING $f"
    fi
done
