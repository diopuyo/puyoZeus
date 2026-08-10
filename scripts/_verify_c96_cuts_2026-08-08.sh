#!/usr/bin/env bash
# c96 切り出し 3 本の検収: duration 確認 + 中間地点フレームの静止画抽出。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
OUTDIR="$ROOT/data/verify/c96_split_2026-08-08/cuts"
mkdir -p "$OUTDIR"
FFMPEG=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
for i in 1 2 3; do
  F="$HOME/frames/video_c96s${i}.mp4"
  [ -f "$F" ] || { echo "[verify] MISSING $F"; continue; }
  # duration は ffmpeg のメタ出力から拾う (ffprobe はバンドルに無いため)
  DUR=$("$FFMPEG" -nostdin -hide_banner -i "$F" 2>&1 | grep -m1 Duration)
  echo "[verify] s${i} size=$(stat -c %s "$F") $DUR"
  # 中間・序盤・終盤の 3 点でフレーム抽出 (試合画面が写っているか目視用)
  for T in 60 900 1800; do
    "$FFMPEG" -nostdin -hide_banner -loglevel error -y -ss "$T" -i "$F" \
      -frames:v 1 "$OUTDIR/s${i}_t${T}.png"
  done
done
echo "[verify] ALL_DONE  -> $OUTDIR"
