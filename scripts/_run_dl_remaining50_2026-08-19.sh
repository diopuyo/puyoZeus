#!/bin/bash
# 50本セットのうちDLに失敗した分を、収集と並行して全部取得する。
# 収集本体は一度失敗した target を再訪しない設計なので二重DLは起きない。
# Cookie は毎回マスターから使い捨てコピーを作る (yt-dlp の書き戻しでマスターが
# 壊れるのを防ぐ、2026-08-18 実測)。直列で回す。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
mkdir -p logs
exec >> logs/dl_remaining50_2026-08-19.log 2>&1
echo "[dl_rem] 開始 $(date)"
SP="/mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/22abd085-8e57-4d2a-857e-8516be642774/scratchpad"
CKM="$SP/yt_cookies_master.txt"
CK="$SP/yt_cookies_work_rem50.txt"
NODE=/home/ryouj/.nvm/versions/node/v24.19.0/bin/node
FF=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin
FMT='bv*[vcodec^=avc1][height<=1080]+ba/b[ext=mp4][vcodec^=avc1][height<=1080]/b[height<=1080][vcodec!*=av01]/b[ext=mp4]'
MAN=data/verify/regen_2026-08-19_subset50/manifest.tsv
ST=data/verify/regen_2026-08-19_subset50/status.tsv
for round in 1 2 3; do
  echo "[dl_rem] 第${round}周 $(date +%H:%M:%S)"
  ok=0; ng=0
  while IFS=$'\t' read -r tid fname vid tier origin; do
    [ "$tid" = "target_id" ] && continue
    OUT="data/frames/$fname"
    [ -s "$OUT" ] && continue
    # status上でDL失敗しているものだけ対象 (収集中のものは触らない)
    grep -qP "^${tid}\t.*SKIP_DL_FAIL" "$ST" 2>/dev/null || continue
    for attempt in 0 1 2 3 4; do
      cp "$CKM" "$CK" 2>/dev/null; chmod 644 "$CK" 2>/dev/null
      nice -n 15 ./venv/bin/python -m yt_dlp --ffmpeg-location "$FF" \
        --js-runtimes "node:$NODE" --cookies "$CK" \
        -f "$FMT" --remux-video mp4 --no-playlist --no-progress \
        -o "$OUT" "https://www.youtube.com/watch?v=$vid" 2>&1 | grep -oE 'ERROR.*' | head -1
      [ -s "$OUT" ] && break
      rm -f data/frames/${fname}*.part 2>/dev/null
      sleep 60
    done
    if [ -s "$OUT" ]; then echo "[dl_rem][$tid] OK $(date +%H:%M:%S)"; ok=$((ok+1));
    else echo "[dl_rem][$tid] 失敗 $(date +%H:%M:%S)"; ng=$((ng+1)); fi
    sleep 30
  done < "$MAN"
  echo "[dl_rem] 第${round}周 完了 成功=$ok 失敗=$ng"
  [ "$ng" -eq 0 ] && break
  sleep 900
done
echo "[dl_rem] 終了 $(date)"
