#!/usr/bin/env bash
# 「120秒を超える試合」が本当に1試合か、WIN★パネルの数字で確かめる。
# 試合の開始直後・中間・終了直前のフレームを抜き、勝利数が途中で変わっていれば
# 境界検出の失敗 (= 複数試合が1つに繋がっている)。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
OUT=data/verify/long_match_check
mkdir -p "$OUT"
FF=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
# vid:start:mid:end  (long_matches_2026-08-09.tsv より)
PAIRS="c54:5:100:198 c44:5:100:197 c32:5:100:197"
for P in $PAIRS; do
  VID=$(echo $P | cut -d: -f1); S=$(echo $P | cut -d: -f2)
  M=$(echo $P | cut -d: -f3); E=$(echo $P | cut -d: -f4)
  SRC="$HOME/frames/video_${VID}.mp4"
  [ -f "$SRC" ] || { echo "skip $VID"; continue; }
  for T in $S $M $E; do
    "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" -i "$SRC"       -frames:v 1 "$OUT/${VID}_t${T}.png"
  done
  echo "[check] $VID : t=$S $M $E"
done
ls "$OUT"
