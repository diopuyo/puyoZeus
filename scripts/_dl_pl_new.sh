#!/bin/bash
# 新おいうリーグ・チャレンジャー30本(pl_new_missing.tsv)をH.264強制でDL。video_c4..c33。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin
i=4
while IFS=$'\t' read -r id dur title; do
  [ -z "$id" ] && continue
  out="data/frames/video_c$i.mp4"
  if [ -s "$out" ]; then echo "[skip] video_c$i exists"; i=$((i+1)); continue; fi
  echo "[DL] video_c$i <- $id ($title)"
  rm -f "data/frames/video_c$i".*.part "data/frames/video_c$i.mp4"
  nice -n 15 ./venv/bin/python -m yt_dlp --ffmpeg-location "$FF" \
    -f 'bv*[vcodec^=avc1][height<=1080]+ba/b[ext=mp4][vcodec^=avc1][height<=1080]/b[height<=1080][vcodec!*=av01]/b[ext=mp4]' \
    --remux-video mp4 --no-playlist --no-progress -o "$out" \
    "https://www.youtube.com/watch?v=$id" 2>&1 | tail -1
  i=$((i+1))
done < <(tail -n +2 data/pl_new_missing.tsv)
echo "[DL] done. video_c*.mp4 total:"; ls -1 data/frames/video_c*.mp4 2>/dev/null | wc -l
