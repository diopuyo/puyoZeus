#!/usr/bin/env bash
# サイクル66 並列検証パック
# v91 viz + v89 eval + v91 eval + 多 frame diag を一気に走らせる
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
mkdir -p logs

# 1. v91 viz (= 主検証対象)
setsid -f bash -c 'PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video data/test_unknown/v91_match1_75s_720p.mp4 \
    --output data/test_unknown/v91_match1_75s_viz_FIX_O.mp4 \
    > logs/viz_v91_match1_FIX_O.log 2>&1 < /dev/null'

# 2. v89 regression eval
setsid -f bash -c 'PYTHONPATH=. ./venv/bin/python -m scripts.eval_landing_via_next_history \
    --video data/evaluation_videos/v89_match3_95s.mp4 \
    --hsv-state data/per_video_hsv_ranges/v89.json \
    --max-sec 90 \
    --out-json data/eval_results/landing_v89_FIX_O.json \
    > logs/eval_v89_FIX_O.log 2>&1 < /dev/null'

# 3. v91 eval
setsid -f bash -c 'PYTHONPATH=. ./venv/bin/python -m scripts.eval_landing_via_next_history \
    --video data/test_unknown/v91_match1_75s_720p.mp4 \
    --max-sec 70 \
    --out-json data/eval_results/landing_v91_match1_FIX_O.json \
    > logs/eval_v91_FIX_O.log 2>&1 < /dev/null'

# 4. v40 regression eval
setsid -f bash -c 'PYTHONPATH=. ./venv/bin/python -m scripts.eval_landing_via_next_history \
    --video data/evaluation_videos/v40_match7_125s.mp4 \
    --hsv-state data/per_video_hsv_ranges/v40.json \
    --max-sec 120 \
    --out-json data/eval_results/landing_v40_FIX_O.json \
    > logs/eval_v40_FIX_O.log 2>&1 < /dev/null'

echo "all 4 jobs started"
