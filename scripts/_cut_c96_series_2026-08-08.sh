#!/usr/bin/env bash
# video_c96 (5.5時間・3シリーズ連結) を series_segments.tsv に従い 3 本へ切り出す。
# 再エンコードなし (-c copy) なので実時間は数分。出力は $HOME/frames/video_c96s{1,2,3}.mp4。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
SRC="$HOME/frames/_hold_video_c96.mp4"
TSV="$ROOT/data/verify/c96_split_2026-08-08/series_segments.tsv"
FFMPEG=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
echo "[cut] ffmpeg=$FFMPEG"
# TSV: series_idx / clip_start_sec / clip_end_sec / ... (ヘッダ1行)
# 注意: ffmpeg は stdin を読むため -nostdin 必須。付けないと while read の
# 入力を食って行がずれる (2026-08-08 に実際に発生し s2/s3 が壊れた)。
# 対象 series は第1引数で絞れる (未指定なら全件)。
ONLY="${1:-}"
tail -n +2 "$TSV" | while IFS=$'\t' read -r idx start end rest; do
  if [ -n "$ONLY" ] && [ "$idx" != "$ONLY" ]; then continue; fi
  OUT="$HOME/frames/video_c96s${idx}.mp4"
  echo "[cut] series $idx: $start -> $end  => $OUT"
  "$FFMPEG" -nostdin -hide_banner -loglevel error -y -ss "$start" -to "$end" -i "$SRC" -c copy "$OUT"
  echo "[cut] done series $idx rc=$? size=$(stat -c %s "$OUT" 2>/dev/null)"
done
echo "[cut] ALL_DONE"
