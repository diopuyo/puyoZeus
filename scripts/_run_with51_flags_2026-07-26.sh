#!/usr/bin/env bash
# #51 フラグ (recovery_counter_carryover + cnn_flicker_hsv_fallback) ON構成で
# 19clip cell正解率を再測定する。baseline (current_default_2026-07-26.json) と
# 完全同一の動画リスト・設定 (workers=3, sample-interval=0.06666666,
# --game-event-chain-exit --landing-observed-color --no-per-video-hsv) に
# #51 フラグ2種のみ追加する。
set -euo pipefail
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

VIDEOS="v29_match2_156s,v40_match7_125s,v51_match2_97s,v57_match2_100s,v70_match2_113s,v89_match3_95s,v95_match15_99s,v97_match11_96s,v29m2_buf15s,v30_5min_90s,v30_match11_89s,v30_match11_buf15s,v40m7_buf15s,v51m2_buf15s,v57m2_buf15s,v70m2_buf15s,v89m3_buf15s,v95m15_buf15s,v97_match11_buf15s"

PYTHONPATH=. ./venv/bin/python scripts/measure_stable_cell_acc.py \
  --videos "$VIDEOS" \
  --sample-interval 0.06666666 \
  --workers 3 \
  --game-event-chain-exit \
  --landing-observed-color \
  --no-per-video-hsv \
  --recovery-counter-carryover \
  --cnn-flicker-hsv-fallback \
  --output data/verify/cell_accuracy_recheck_2026-07-26/with51_flags_2026-07-26.json
