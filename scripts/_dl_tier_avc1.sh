#!/bin/bash
# S級3+チャレンジャー3を H.264(avc1)強制で再DL。
# 前回 AV1 で落ちた(OpenCVがAV1デコード不可)。avc1ならOpenCVで確実に読める。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin
names=(s1 s2 s3 c1 c2 c3)
ids=(04Lb9BZCpP0 UpnGj22itdA GDfVPnyrfwU kETyIUk_Vb8 Lw6Z8Nzguo4 fIIDFPCD2w0)
for i in "${!names[@]}"; do
  n="${names[$i]}"; id="${ids[$i]}"
  rm -f "data/frames/video_$n.mp4" "data/frames/video_$n".*.part
  echo "[DL-avc1] video_$n <- $id"
  nice -n 15 ./venv/bin/python -m yt_dlp --ffmpeg-location "$FF" \
    -f 'bv*[vcodec^=avc1][height<=1080]+ba/b[ext=mp4][vcodec^=avc1][height<=1080]/b[height<=1080][vcodec!*=av01]/b[ext=mp4]' \
    --remux-video mp4 --no-playlist --no-progress \
    -o "data/frames/video_$n.%(ext)s" \
    "https://www.youtube.com/watch?v=$id" 2>&1 | tail -2
done
echo "[DL-avc1] done"
# コーデック確認
for n in s1 s2 s3 c1 c2 c3; do
  echo -n "video_$n: "
  "$FF/ffmpeg" -i "data/frames/video_$n.mp4" 2>&1 | grep -oE 'Video: [a-z0-9]+' | head -1
done
