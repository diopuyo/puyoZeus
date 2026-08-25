#!/bin/bash
# 30先2セットの動画 (全知全能ぷよ、1時間57分) を取得する。
# user 依頼: この動画で有利不利指標入りの解析動画を作る (30先ごとに2本に分割)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
mkdir -p logs
exec >> logs/dl_zenchi_2026-08-19.log 2>&1
echo "[zenchi] 開始 $(date)"
SP="/mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/22abd085-8e57-4d2a-857e-8516be642774/scratchpad"
CKM="$SP/yt_cookies_master.txt"
CK="$SP/yt_cookies_work_zenchi.txt"
NODE=/home/ryouj/.nvm/versions/node/v24.19.0/bin/node
FF=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin
FMT='bv*[vcodec^=avc1][height<=1080]+ba/b[ext=mp4][vcodec^=avc1][height<=1080]/b[height<=1080][vcodec!*=av01]/b[ext=mp4]'
OUT=data/frames/video_zenchi_c0BQoMJwwQU.mp4
for attempt in 0 1 2 3 4 5; do
  [ -s "$OUT" ] && break
  cp "$CKM" "$CK" 2>/dev/null; chmod 644 "$CK" 2>/dev/null
  echo "[zenchi] attempt=$attempt $(date +%H:%M:%S)"
  nice -n 15 ./venv/bin/python -m yt_dlp --ffmpeg-location "$FF" \
    --js-runtimes "node:$NODE" --cookies "$CK" \
    -f "$FMT" --remux-video mp4 --no-playlist --no-progress \
    -o "$OUT" "https://www.youtube.com/watch?v=c0BQoMJwwQU" 2>&1 | grep -oE 'ERROR.*' | head -1
  [ -s "$OUT" ] && break
  rm -f data/frames/video_zenchi*.part 2>/dev/null
  sleep 60
done
if [ -s "$OUT" ]; then echo "[zenchi] 完了 $(ls -la "$OUT" | awk '{printf "%.2fGB", $5/1073741824}')"; else echo "[zenchi] 取得失敗"; fi
echo "[zenchi] 終了 $(date)"
