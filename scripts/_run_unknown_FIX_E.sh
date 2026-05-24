#!/usr/bin/env bash
# unknown1-5 で FIX_E viz を並列生成 (slide motion 抑制版)
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
mkdir -p logs
for tag in "" 2 3 4 5; do
    video=data/test_unknown/unknown${tag}_match_120s.mp4
    out=data/test_unknown/unknown${tag}_viz_FIX_E.mp4
    log=logs/viz_unknown${tag}_FIX_E.log
    setsid -f bash -c "PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition --video $video --output $out --hsv-state data/per_video_hsv_ranges/_merged_default.json > $log 2>&1 < /dev/null"
    echo "started: unknown${tag} -> $out"
done
echo all_started
