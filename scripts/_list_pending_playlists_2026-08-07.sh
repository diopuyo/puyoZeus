#!/bin/bash
# plB/plD の残り動画 + チャンネル新着プレイリスト候補のタイトルを列挙 (ティア確認用)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
OUT="data/_pending_dl_candidates_2026-08-07.tsv"
: > "$OUT"
echo "=== plB (全12本、既DL=先頭6) ==="
./venv/bin/python -m yt_dlp --flat-playlist \
  --print "%(playlist_index)s\t%(id)s\t%(duration)s\t%(title)s" \
  "https://www.youtube.com/playlist?list=PLsjREVssD8baOyWw8zpRqV0ru42Cy52Ik" | tee -a "$OUT"
echo "=== plD (全7本、既DL=先頭6) ==="
./venv/bin/python -m yt_dlp --flat-playlist \
  --print "%(playlist_index)s\t%(id)s\t%(duration)s\t%(title)s" \
  "https://www.youtube.com/playlist?list=PLsjREVssD8bYG_VUIlJvREnco92HB5R3t" | tee -a "$OUT"
echo "[done] -> $OUT"
