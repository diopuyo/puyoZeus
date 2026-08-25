#!/bin/bash
# 50本セットの不足分 (マスター級10本) をDLする。
# Cookie は単一ファイルを直接渡し、yt-dlp によるセッション書き戻しを許可する
# (2026-08-18: コピーを渡すと更新が捨てられ、元Cookieがローテーションで
#  無効化されて403に戻る事象を実測)。そのため必ず直列で回す。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
mkdir -p logs
# 148本体と同じ方式: スクリプト自身がログへ追記する (外部リダイレクトだと
# setsid -f 経由で起動しない事象を実測、2026-08-18)
exec >> logs/dl_subset50_2026-08-18.log 2>&1
echo "[dl50] 開始 $(date)"
SP="/mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/22abd085-8e57-4d2a-857e-8516be642774/scratchpad"
CKM="$SP/yt_cookies_master.txt"
CK="$SP/yt_cookies_work_dl50.txt"
NODE=/home/ryouj/.nvm/versions/node/v24.19.0/bin/node
FF=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin
FMT='bv*[vcodec^=avc1][height<=1080]+ba/b[ext=mp4][vcodec^=avc1][height<=1080]/b[height<=1080][vcodec!*=av01]/b[ext=mp4]'
PROG=logs/dl_subset50_progress.txt
: > "$PROG"
ok=0; ng=0
while IFS=$'\t' read -r tid fname vid tier origin local; do
  [ "$tid" = "target_id" ] && continue
  [ "$local" = "Y" ] && continue
  OUT="data/frames/$fname"
  if [ -s "$OUT" ]; then echo "[dl50][$tid] 既存 skip" >> "$PROG"; ok=$((ok+1)); continue; fi
  for attempt in 0 1 2 3 4 5; do
    # yt-dlp は --cookies に渡したファイルへセッションを書き戻し、その際に
    # YouTube以外のCookieを削除してログイン情報を壊す (2026-08-18 実測:
    # 2463行/LOGIN_INFO 2件 -> 1140行/0件 に退化し403が再発)。
    # マスターは触らせず、毎回使い捨てコピーを渡す。
    cp "$CKM" "$CK" 2>/dev/null; chmod 644 "$CK" 2>/dev/null
    echo "[dl50][$tid] attempt=$attempt $(date +%H:%M:%S)" >> "$PROG"
    nice -n 10 ./venv/bin/python -m yt_dlp --newline --ffmpeg-location "$FF" \
      --js-runtimes "node:$NODE" --cookies "$CK" \
      -f "$FMT" --remux-video mp4 --no-playlist --no-progress \
      -o "$OUT" "https://www.youtube.com/watch?v=$vid" >> "$PROG" 2>&1
    if [ -s "$OUT" ]; then echo "[dl50][$tid] OK $(date +%H:%M:%S)" >> "$PROG"; ok=$((ok+1)); break; fi
    rm -f data/frames/${fname}*.part 2>/dev/null
    sleep 45
  done
  if [ ! -s "$OUT" ]; then echo "[dl50][$tid] 断念 $(date +%H:%M:%S)" >> "$PROG"; ng=$((ng+1)); fi
  sleep 20
done < data/verify/subset50_targets_2026-08-18.tsv
echo "[dl50] 完了 成功=$ok 失敗=$ng $(date)" >> "$PROG"
