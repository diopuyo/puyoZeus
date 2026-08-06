#!/bin/bash
# data/_dl_expand2_2026-08-07.tsv (name<TAB>id<TAB>title) を H.264強制で直列DL。
# 既存はskip。scripts/_dl_expand.sh (Phase L step1) のパターンを踏襲、
# 読み込み元のみ新TSVに変更 (Phase L step2、S級/A級補強、2026-08-07)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin
ok=0; skip=0; fail=0
while IFS=$'\t' read -r name id rest; do
  [ -z "$id" ] && continue
  out="data/frames/$name.mp4"
  if [ -s "$out" ]; then echo "[skip] $name exists"; skip=$((skip+1)); continue; fi
  rm -f "data/frames/$name".*.part "$out"
  echo "[DL] $name <- $id"
  nice -n 15 ./venv/bin/python -m yt_dlp --ffmpeg-location "$FF" \
    -f 'bv*[vcodec^=avc1][height<=1080]+ba/b[ext=mp4][vcodec^=avc1][height<=1080]/b[height<=1080][vcodec!*=av01]/b[ext=mp4]' \
    --remux-video mp4 --no-playlist --no-progress -o "$out" \
    "https://www.youtube.com/watch?v=$id" > /dev/null 2>&1
  if [ -s "$out" ]; then ok=$((ok+1)); else echo "[FAIL] $name"; fail=$((fail+1)); fi
done < data/_dl_expand2_2026-08-07.tsv
echo "[DL] DONE ok=$ok skip=$skip fail=$fail  total mp4:"; ls -1 data/frames/video_c*.mp4 2>/dev/null | wc -l
