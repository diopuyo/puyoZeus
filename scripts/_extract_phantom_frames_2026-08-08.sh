#!/usr/bin/env bash
# 幻満杯盤面 検知器の妥当性確認: 検知上位動画の代表時刻フレームを抽出する。
# 演出画面/ロビーなら真陽性、 実戦の終局盤面なら偽陽性 (閾値が緩い)。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
OUT=data/verify/phase_l_quality_gate_2026-08-07/triage_frames
mkdir -p "$OUT"
FF=$(./venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
PAIRS="c28:84.57 32:102.43 c117:0 c84:0"
for P in $PAIRS; do
  VID="${P%%:*}"; T="${P##*:}"
  if [ "$T" = "0" ]; then
    # 代表時刻が未指定の動画は TSV から 1 件目を拾う
    T=$(grep -E "^${VID}	" \
      data/verify/phase_l_quality_gate_2026-08-07/phantom_boards_2026-08-08.tsv \
      | sed 's/.*t=\([0-9.]*\).*/\1/' | head -1)
  fi
  SRC="$HOME/frames/video_${VID}.mp4"
  [ -f "$SRC" ] || SRC="$ROOT/data/frames/video_${VID}.mp4"
  if [ ! -f "$SRC" ]; then echo "[phantom] MISSING ${VID}"; continue; fi
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" -i "$SRC" \
    -frames:v 1 "$OUT/phantom_${VID}_t${T}.png" && echo "[phantom] ok ${VID} t=${T}"
done
