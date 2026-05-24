#!/bin/bash
# cycle 32 I1: 8 動画分の seed 抽出を並列 3 で実行
# - skip_ojama=True (default) で ojama 採取スキップ
# - G1 (bg_fp) + G9 (end_skip 180) + G10 (ojama V上限、 ojama skip でも互換維持) 適用
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

# 既存の v89m3 を退避 (前 cycle の古い seed)
if [ -d data/pseudo_labels_hsv_seed/v89m3 ]; then
  mv data/pseudo_labels_hsv_seed/v89m3 data/pseudo_labels_hsv_seed/v89m3_old_$(date +%s) 2>/dev/null || true
fi

run_one() {
  local vid="$1"
  local video_id="$2"
  local log="logs/cycle_32c_seed_${video_id}.log"
  echo "[start] ${video_id} → ${log}"
  PYTHONPATH=. ./venv/bin/python -m scripts.extract_hsv_seed_dataset \
    --video "data/evaluation_videos/${vid}" \
    --video-id "${video_id}" \
    --max-per-color 1500 \
    > "${log}" 2>&1
  echo "[done] ${video_id}"
}

export -f run_one

# 8 動画リスト: video_filename:video_id
cat > /tmp/cycle32_videos.txt <<EOF
v29_match2_156s.mp4 v29m2
v40_match7_125s.mp4 v40m7
v51_match2_97s.mp4 v51m2
v57_match2_100s.mp4 v57m2
v70_match2_113s.mp4 v70m2
v89_match3_95s.mp4 v89m3
v95_match15_99s.mp4 v95m15
v97_match11_96s.mp4 v97m11
EOF

# 並列 3 で実行 (memory project_cycle_findings_2026-05-15.md ルール厳守)
cat /tmp/cycle32_videos.txt | xargs -L 1 -P 3 bash -c 'run_one "$0" "$1"'

echo "=== ALL DONE @ $(date) ===" | tee logs/cycle_32c_all_done.flag
