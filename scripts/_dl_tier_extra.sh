#!/bin/bash
# S級3本 + チャレンジャー3本をDL(較正のティア多様化)。
# yt-dlpのremux用にimageio-ffmpeg同梱バイナリをffmpegとしてsymlink(システムffmpeg無いため)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
BIN="$(pwd)/venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
ln -sf "$BIN" "$(pwd)/venv/bin/ffmpeg"
FF="$(pwd)/venv/bin"
names=(s1 s2 s3 c1 c2 c3)
ids=(04Lb9BZCpP0 UpnGj22itdA GDfVPnyrfwU kETyIUk_Vb8 Lw6Z8Nzguo4 fIIDFPCD2w0)
for i in "${!names[@]}"; do
  n="${names[$i]}"; id="${ids[$i]}"
  echo "[DL] video_$n <- $id"
  nice -n 15 ./venv/bin/python -m yt_dlp --ffmpeg-location "$FF" \
    -f 'bv*[height<=1080]+ba/b[height<=1080]/b' --remux-video mp4 \
    --no-playlist --no-progress -o "data/frames/video_$n.%(ext)s" \
    "https://www.youtube.com/watch?v=$id" 2>&1 | tail -3
done
echo "[DL] done"
ls -lh data/frames/video_s*.mp4 data/frames/video_c*.mp4 2>/dev/null
