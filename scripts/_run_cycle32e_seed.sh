#!/bin/bash
# cycle 32e (2026-05-19): EMPTY 採取追加 (= --max-empty 500)、
# 既存 5 色 seed は cycle 32c のものを再利用するため、 EMPTY のみ追加採取。
# 既存 cycle 32c seed (= v29m2/v40m7/v51m2/v57m2/v70m2/v89m3/v95m15/v97m11) に
# EMPTY label sample を append する形で進める。
#
# 効率化のため、 既存 cell.jsonl を退避 (= cycle 32c 用) して
# 新規 cell.jsonl (= cycle 32e 用、 EMPTY+5色) を作る。
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

VIDEOS=(
  "v29_match2_156s.mp4:v29m2"
  "v40_match7_125s.mp4:v40m7"
  "v51_match2_97s.mp4:v51m2"
  "v57_match2_100s.mp4:v57m2"
  "v70_match2_113s.mp4:v70m2"
  "v89_match3_95s.mp4:v89m3"
  "v95_match15_99s.mp4:v95m15"
  "v97_match11_96s.mp4:v97m11"
)

# 既存 cycle 32c seed を _32c 接尾子で archive
for entry in "${VIDEOS[@]}"; do
  vid_id="${entry##*:}"
  src="data/pseudo_labels_hsv_seed/${vid_id}/cell.jsonl"
  dst="data/pseudo_labels_hsv_seed/${vid_id}/cell_32c.jsonl"
  if [ -f "$src" ] && [ ! -f "$dst" ]; then
    cp "$src" "$dst"
    echo "[backup] ${vid_id} cell.jsonl → cell_32c.jsonl"
  fi
done

run_one() {
  local vid="$1"
  local video_id="$2"
  local log="logs/cycle_32e_seed_${video_id}.log"
  echo "[start] ${video_id} → ${log}"
  # 既存 cell.jsonl を削除して新規採取 (= 5色 + EMPTY 一括)
  rm -f "data/pseudo_labels_hsv_seed/${video_id}/cell.jsonl"
  PYTHONPATH=. ./venv/bin/python -m scripts.extract_hsv_seed_dataset \
    --video "data/evaluation_videos/${vid}" \
    --video-id "${video_id}" \
    --max-per-color 1500 \
    --max-empty 500 \
    > "${log}" 2>&1
  echo "[done] ${video_id}"
}

export -f run_one

cat > /tmp/cycle32e_videos.txt <<EOF
v29_match2_156s.mp4 v29m2
v40_match7_125s.mp4 v40m7
v51_match2_97s.mp4 v51m2
v57_match2_100s.mp4 v57m2
v70_match2_113s.mp4 v70m2
v89_match3_95s.mp4 v89m3
v95_match15_99s.mp4 v95m15
v97_match11_96s.mp4 v97m11
EOF

# 並列 3 (memory cycle ルール厳守)
cat /tmp/cycle32e_videos.txt | xargs -L 1 -P 3 bash -c 'run_one "$0" "$1"'

echo "=== ALL DONE @ $(date) ===" | tee logs/cycle_32e_seed_done.flag
