#!/bin/bash
# 148本のうち手元に無い107本を、今後の収集に備えて先に取得しておく。
# ティア優先順 (マスター>チャレンジャー>S級>A級) で回す。
#
# 安全弁: 空き容量が MIN_FREE_GB を下回ったら待機する (推定160GB必要に対し
# 空きは約199GBでぎりぎりのため)。収集が動画を消費・削除して空きが戻る。
# Cookie は毎回マスターから使い捨てコピーを作る (yt-dlp の書き戻しでマスターが
# 壊れるのを防ぐ、2026-08-18 実測)。直列で回す。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
mkdir -p logs
exec >> logs/dl_all148_2026-08-19.log 2>&1
echo "[dl148] 開始 $(date)"
SP="/mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/22abd085-8e57-4d2a-857e-8516be642774/scratchpad"
CKM="$SP/yt_cookies_master.txt"
CK="$SP/yt_cookies_work_all148.txt"
NODE=/home/ryouj/.nvm/versions/node/v24.19.0/bin/node
FF=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin
FMT='bv*[vcodec^=avc1][height<=1080]+ba/b[ext=mp4][vcodec^=avc1][height<=1080]/b[height<=1080][vcodec!*=av01]/b[ext=mp4]'
MAN=data/verify/dl_all148_targets_2026-08-19.tsv
MIN_FREE_GB=40
for round in 1 2 3; do
  echo "[dl148] 第${round}周 $(date +%H:%M:%S)"
  ok=0; ng=0; skip=0
  while IFS=$'\t' read -r tid fname vid tier origin; do
    [ "$tid" = "target_id" ] && continue
    OUT="data/frames/$fname"
    if [ -s "$OUT" ]; then skip=$((skip+1)); continue; fi
    # 空き容量チェック (収集が動画を削除して空きが戻るのを待つ)
    while true; do
      free_gb=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc '0-9')
      [ "${free_gb:-0}" -ge "$MIN_FREE_GB" ] && break
      echo "[dl148] 空き ${free_gb}GB < ${MIN_FREE_GB}GB のため待機 $(date +%H:%M:%S)"
      sleep 600
    done
    for attempt in 0 1 2 3; do
      cp "$CKM" "$CK" 2>/dev/null; chmod 644 "$CK" 2>/dev/null
      nice -n 19 ./venv/bin/python -m yt_dlp --ffmpeg-location "$FF" \
        --js-runtimes "node:$NODE" --cookies "$CK" \
        -f "$FMT" --remux-video mp4 --no-playlist --no-progress \
        -o "$OUT" "https://www.youtube.com/watch?v=$vid" 2>&1 | grep -oE 'ERROR.*' | head -1
      [ -s "$OUT" ] && break
      rm -f data/frames/${fname}*.part 2>/dev/null
      sleep 60
    done
    if [ -s "$OUT" ]; then echo "[dl148][$tid] OK ($tier) $(date +%H:%M:%S)"; ok=$((ok+1));
    else echo "[dl148][$tid] 失敗 ($tier)"; ng=$((ng+1)); fi
    sleep 20
  done < "$MAN"
  echo "[dl148] 第${round}周 完了 成功=$ok 失敗=$ng 既存=$skip"
  [ "$ng" -eq 0 ] && break
  sleep 1800
done
echo "[dl148] 終了 $(date)"
