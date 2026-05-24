#!/usr/bin/env bash
# 全既知動画で eval_landing 実行
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
mkdir -p data/eval_results logs

declare -A VIDEOS=(
    [v89]="data/evaluation_videos/v89_match3_95s.mp4 90"
    [v40]="data/evaluation_videos/v40_match7_125s.mp4 120"
    [v29]="data/evaluation_videos/v29_match2_156s.mp4 150"
    [v51]="data/evaluation_videos/v51_match2_97s.mp4 90"
    [v57]="data/evaluation_videos/v57_match2_100s.mp4 95"
    [v70]="data/evaluation_videos/v70_match2_113s.mp4 110"
)

for tag in v89 v40 v29 v51 v57 v70; do
    info=${VIDEOS[$tag]}
    video=${info% *}
    max_sec=${info#* }
    out=data/eval_results/landing_${tag}_FIX_L.json
    echo "=== $tag ($video, max=${max_sec}s) ==="
    PYTHONPATH=. ./venv/bin/python -m scripts.eval_landing_via_next_history \
        --video "$video" \
        --hsv-state data/per_video_hsv_ranges/${tag}.json \
        --max-sec "$max_sec" \
        --out-json "$out" 2>&1 | tail -5
done
echo "all done"
