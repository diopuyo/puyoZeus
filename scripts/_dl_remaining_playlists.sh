#!/bin/bash
# 残り 55 本を一括 DL するスクリプト (video_41 から連番)
# plB: idx 7-12 (6本)、plC: idx 7-54 (48本)、plD: idx 7 (1本)
#
# 利用例:
#     bash scripts/_dl_remaining_playlists.sh
# 完了後は scripts/phase_e_count_match や detect_match_winners も
# v41-v95 (-2 = v39, v22 等の異常をスキップした連番) について実行する必要あり

cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
LOG=data/phase_e_dl_remaining.log
> "$LOG"

PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_dl_playlists \
    --start-index 41 --max-per-playlist 0 --min-duration 600 \
    --playlists plB,plC,plD \
    --skip-playlist-idx 'plB:1-6,plC:1-6,plD:1-6' \
    --parallel 4 2>&1 | tee -a "$LOG"

# DL 完了後の後続処理:
#   1. count_match_v4 で v41-v?? の matches.tsv 生成
#   2. detect_match_winners で winners.tsv 生成
#   3. phase_e_collect_indicator_dataset で v41-v?? を shard として生成
#   4. shard 統合 + learn_weights_v3 / phase_e_learn_phase_aware で再学習
