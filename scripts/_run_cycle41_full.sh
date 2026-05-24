#!/bin/bash
# cycle 41 残り 2 動画 (v89m3, v70m2) viz 生成
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

for entry in "v89_match3_95s.mp4:v89m3" "v70_match2_113s.mp4:v70m2"; do
  vid_file="${entry%%:*}"
  vid_id="${entry##*:}"
  echo "[viz-start] ${vid_id}"
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "data/evaluation_videos/${vid_file}" \
    --output "data/review_videos/cycle41/cycle41_${vid_id}.mp4" \
    --cnn-model models/cnn_phase_b_large_v2.pt \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "logs/board_logs/cycle41_${vid_id}.jsonl" \
    --use-puyo-gate \
    > "logs/cycle_41_viz_${vid_id}.log" 2>&1
  echo "[viz-done] ${vid_id} @ $(date)"
done

for vid_id in v89m3 v70m2; do
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "logs/board_logs/cycle41_${vid_id}.jsonl" \
    --report-out "data/verify/cycle41_eval/cycle41_${vid_id}.json" \
    > "logs/cycle_41_eval_${vid_id}.log" 2>&1
done

echo "=== cycle 41 FULL DONE @ $(date) ===" | tee logs/cycle_41_full_done.flag
