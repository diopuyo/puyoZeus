#!/usr/bin/env bash
# 古い FIX_A〜FIX_N viz + 360p unknown + 古い compare image を削除
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

echo === before ===
du -sh data/test_unknown data/evaluation_videos 2>&1

# 360p unknown を全削除 (スコープ外)
rm -f data/test_unknown/unknown*_match*.mp4
rm -f data/test_unknown/unknown*_viz_*.mp4
# 旧 FIX 系 (A-N) v91 viz
rm -f data/test_unknown/v91_match1_75s_viz_FIX_[A-N].mp4
# evaluation_videos の中間 viz (FIX_A-N)
rm -f data/evaluation_videos/v89_match3_phase_i_viz_FIX_[A-N].mp4
rm -f data/evaluation_videos/v40_phase_i_viz_FIX_[A-N].mp4
rm -f data/evaluation_videos/v89_match3_phase_i_viz_FINAL2.mp4
rm -f data/evaluation_videos/v40_phase_i_viz_FINAL2.mp4
rm -f data/evaluation_videos/v89_match3_phase_i_viz_MERGED.mp4
rm -f data/evaluation_videos/v89_match3_phase_i_viz_baseline.mp4
rm -f data/evaluation_videos/v29_phase_i_viz_FINAL2.mp4
rm -f data/evaluation_videos/v29_phase_i_viz_FIX_A.mp4
rm -f data/evaluation_videos/v51_phase_i_viz_FINAL2.mp4
rm -f data/evaluation_videos/v57_phase_i_viz_FINAL2.mp4
rm -f data/evaluation_videos/v70_phase_i_viz_FINAL2.mp4

# 古い compare image
rm -rf data/evaluation_videos/v29_compare
rm -rf data/evaluation_videos/v40_compare
rm -rf data/evaluation_videos/v89_compare
rm -rf data/test_unknown/compare

echo === after ===
du -sh data/test_unknown data/evaluation_videos 2>&1
ls data/test_unknown 2>&1
