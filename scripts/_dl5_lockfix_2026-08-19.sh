#!/bin/bash
# ラッチ修正検証用の再DL 5本 (2026-08-19)。高lock 4本 + 健常対照 1本。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
MASTER="/mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/22abd085-8e57-4d2a-857e-8516be642774/scratchpad/yt_cookies_master.txt"
NODE24="/home/ryouj/.nvm/versions/node/v24.19.0/bin/node"
FFMPEG="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin"
dl() {
  local stem="$1" ytid="$2"
  local out="data/frames/video_${stem}.mp4"
  if [ -s "$out" ]; then echo "[dl] skip $stem (exists)"; return 0; fi
  local work="/tmp/yt_cookies_work_lockfix_${stem}.txt"
  cp "$MASTER" "$work" 2>/dev/null && chmod 644 "$work"
  local cargs=""
  [ -s "$work" ] && cargs="--cookies $work"
  for attempt in 1 2 3; do
    ./venv/bin/python -m yt_dlp --ffmpeg-location "$FFMPEG" \
      --js-runtimes "node:${NODE24}" $cargs \
      -f "bv*[vcodec^=avc1][height<=1080]+ba/b[ext=mp4][vcodec^=avc1][height<=1080]/b[height<=1080][vcodec!*=av01]/b[ext=mp4]" \
      --remux-video mp4 --no-playlist --no-progress \
      -o "$out" "https://www.youtube.com/watch?v=${ytid}" && break
    rm -f "$out"*part
    sleep 45
  done
  [ -s "$out" ] && echo "[dl] OK $stem" || echo "[dl] FAIL $stem"
}
dl c13 sxIU0Cr_iSQ
dl c113 kOWy50IddfI
dl c111 d1i38E4gE1s
dl c135 DWllSSML7YY
dl c11 AccHEU_5024
echo "[dl] all done"
