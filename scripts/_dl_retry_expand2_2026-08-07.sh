#!/bin/bash
# _dl_expand2 のFAIL4本をエラー可視化付きで再試行 (原因判定用にstderr表示)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin
names=(c104 c113 c123 c141)
ids=(-w9ziMGve40 kOWy50IddfI TkvXHB-Bqcw R7vU1wQ4BxI)
for i in "${!names[@]}"; do
  n="${names[$i]}"; id="${ids[$i]}"
  out="data/frames/video_$n.mp4"
  [ -s "$out" ] && { echo "[skip] video_$n exists"; continue; }
  rm -f "data/frames/video_$n".*.part "$out"
  echo "===== [retry] video_$n <- $id ====="
  nice -n 15 ./venv/bin/python -m yt_dlp --no-update --ffmpeg-location "$FF" \
    -f 'bv*[vcodec^=avc1][height<=1080]+ba/b[ext=mp4][vcodec^=avc1][height<=1080]/b[height<=1080][vcodec!*=av01]/b[ext=mp4]' \
    --remux-video mp4 --no-playlist --no-progress -o "$out" \
    "https://www.youtube.com/watch?v=$id" 2>&1 | tail -5
  if [ -s "$out" ]; then echo "[OK] video_$n"; else echo "[STILL-FAIL] video_$n"; fi
done
