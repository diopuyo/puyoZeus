#!/bin/bash
# evaluation_videos_v2 標準化 (= 15s バッファ付き 8 動画 を default に)
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/evaluation_videos_v2
cp data/holdout_videos/v29m2_buf15s.mp4 data/evaluation_videos_v2/v29m2_buf15s.mp4
cp data/holdout_videos/v40m7_buf15s.mp4 data/evaluation_videos_v2/v40m7_buf15s.mp4
cp data/holdout_videos/v51m2_buf15s.mp4 data/evaluation_videos_v2/v51m2_buf15s.mp4
cp data/holdout_videos/v57m2_buf15s.mp4 data/evaluation_videos_v2/v57m2_buf15s.mp4
cp data/holdout_videos/v70m2_buf15s.mp4 data/evaluation_videos_v2/v70m2_buf15s.mp4
cp data/holdout_videos/v89m3_buf15s.mp4 data/evaluation_videos_v2/v89m3_buf15s.mp4
cp data/holdout_videos/v95m15_buf15s.mp4 data/evaluation_videos_v2/v95m15_buf15s.mp4
cp data/holdout_videos/v97m11_buf15s.mp4 data/evaluation_videos_v2/v97m11_buf15s.mp4
echo "=== evaluation_videos_v2 ==="
ls -la data/evaluation_videos_v2/
