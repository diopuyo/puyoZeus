#!/bin/bash
# デコード破損した動画2本 (38/39) を再取得する。
# 39は実長3146秒中1152秒で収集打ち切り、38も1573/3215秒で打ち切り (h264 NALエラー多発)。
# memory feedback_redownload_content_drift_2026-08-14 の再演。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
mkdir -p logs
exec >> logs/redl_broken_2026-08-19.log 2>&1
echo "[redl] 開始 $(date)"
SP="/mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/22abd085-8e57-4d2a-857e-8516be642774/scratchpad"
CKM="$SP/yt_cookies_master.txt"
NODE=/home/ryouj/.nvm/versions/node/v24.19.0/bin/node
FF=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin
FMT='bv*[vcodec^=avc1][height<=1080]+ba/b[ext=mp4][vcodec^=avc1][height<=1080]/b[height<=1080][vcodec!*=av01]/b[ext=mp4]'
for tid in 38 39; do
  vid=$(awk -F'\t' -v t="$tid" '$1==t{print $3}' data/verify/regen_2026-08-11_manifest.tsv)
  OUT="data/frames/video_${tid}.mp4"
  CK="$SP/yt_cookies_work_redl${tid}.txt"
  for attempt in 0 1 2 3; do
    [ -s "$OUT" ] && break
    cp "$CKM" "$CK" 2>/dev/null; chmod 644 "$CK" 2>/dev/null
    echo "[redl][$tid] attempt=$attempt $(date +%H:%M:%S)"
    nice -n 15 ./venv/bin/python -m yt_dlp --ffmpeg-location "$FF" \
      --js-runtimes "node:$NODE" --cookies "$CK" \
      -f "$FMT" --remux-video mp4 --no-playlist --no-progress \
      -o "$OUT" "https://www.youtube.com/watch?v=$vid" 2>&1 | grep -oE 'ERROR.*' | head -1
    [ -s "$OUT" ] && break
    rm -f data/frames/video_${tid}*.part 2>/dev/null
    sleep 60
  done
  # デコード健全性の確認 (cv2で最終フレームまで到達できるか)
  if [ -s "$OUT" ]; then
    ./venv/bin/python -c "
import cv2,sys
c=cv2.VideoCapture('$OUT')
n=c.get(cv2.CAP_PROP_FRAME_COUNT); fps=c.get(cv2.CAP_PROP_FPS)
c.set(cv2.CAP_PROP_POS_FRAMES, max(0,int(n)-30)); ok,_=c.read(); c.release()
print(f'[redl][$tid] 長さ={n/max(fps,1):.0f}秒 末尾読み取り={\"OK\" if ok else \"NG(破損疑い)\"}')
"
  else
    echo "[redl][$tid] 取得失敗"
  fi
done
echo "[redl] 終了 $(date)"
