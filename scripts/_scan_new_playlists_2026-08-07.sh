#!/bin/bash
# 未収集の適格プレイリスト中身を列挙 → data/_new_pl_candidates_2026-08-07.tsv
# 列: pl_tag \t playlist_index \t id \t duration \t title
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
YT="./venv/bin/python -m yt_dlp --no-update"
OUT="data/_new_pl_candidates_2026-08-07.tsv"
: > "$OUT"
scan() {
  local tag="$1" plid="$2"
  echo "=== $tag ($plid) ===" >&2
  $YT --flat-playlist --print "${tag}\t%(playlist_index)s\t%(id)s\t%(duration)s\t%(title)s" \
    "https://www.youtube.com/playlist?list=$plid" >> "$OUT" 2>/dev/null
}
scan r3_alevel   "PLNMQZlSoSRTo"
scan r2_sleague  "PLsjREVssD8bZlPdMhq0kyaVKpSK-Dl7Jk"
scan r2_alevel   "PLsjREVssD8bbd2mMKsf3XHYBADIhitK_a"
scan r1_sleague  "PLsjREVssD8bZcS3TY7BYUOwywpJPuJTh2"
scan r1_aleague  "PLsjREVssD8bZ3X7gtZ23tvPecj3inbfxk"
scan r1_chal     "PLsjREVssD8bbw4ATWhendvoHJP6jtFihy"
scan r1_chal_ket "PLsjREVssD8bbVq8jUW7D01M80oZbChBJ-"
scan r1_master   "PLsjREVssD8bY6jUJbp7CYZT8pS2JBBt6C"
wc -l "$OUT"
