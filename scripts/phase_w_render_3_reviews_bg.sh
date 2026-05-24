#!/bin/bash
# 3 試合 full_review 動画を BG FP 込みで再生成。
set -e
cd "$(dirname "$0")/.."

PY="./venv/bin/python"
OUT="data/verify/phase_w_results"

# (video, start, end, winner, bg_fp_time)
PYTHONPATH=. $PY -m scripts.phase_w_render_full_review \
    --video data/frames/video_04.mp4 --start 9052 --end 9129 \
    --winner 1P --out "$OUT/full_review_v04_m81_bg.mp4" \
    --model models/win_predictor_v3.pt --detect-interval 0.5 \
    --bg-fp-time 9050 \
    2>&1 | tail -5

PYTHONPATH=. $PY -m scripts.phase_w_render_full_review \
    --video data/frames/video_01.mp4 --start 2284 --end 2347 \
    --winner 2P --out "$OUT/full_review_v01_m37_bg.mp4" \
    --model models/win_predictor_v3.pt --detect-interval 0.5 \
    --bg-fp-time 2282 \
    2>&1 | tail -5

PYTHONPATH=. $PY -m scripts.phase_w_render_full_review \
    --video data/frames/video_09.mp4 --start 1954 --end 2015 \
    --winner 1P --out "$OUT/full_review_v09_m28_bg.mp4" \
    --model models/win_predictor_v3.pt --detect-interval 0.5 \
    --bg-fp-time 1952 \
    2>&1 | tail -5

echo "=== complete ==="
ls -la "$OUT"/full_review_*_bg.mp4
