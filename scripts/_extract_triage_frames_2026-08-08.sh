#!/usr/bin/env bash
# 品質ゲート FAIL トリアージ用: 代表時刻の実画面フレームを抽出する。
# 「試合開始+1〜3秒の空盤面」のはずが非空と判定された時刻に、
# 実際は何が写っているか (試合中盤 / 前試合の終局 / 演出画面) を目視確認する。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
OUT=data/verify/phase_l_quality_gate_2026-08-07/triage_frames
mkdir -p "$OUT"
FF=$(./venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
# vid:t_sec の代表ペア (triage_fails_2026-08-08.md の最上位から抜粋)
PAIRS="c12:133.40 c27:1.20 c24:1.17 c57:1278.57 c81:2876.27 36:292.93 c23:265.17"
for P in $PAIRS; do
  VID="${P%%:*}"; T="${P##*:}"
  SRC="$HOME/frames/video_${VID}.mp4"
  if [ ! -f "$SRC" ]; then
    SRC="$ROOT/data/frames/video_${VID}.mp4"
  fi
  if [ ! -f "$SRC" ]; then
    echo "[triage] MISSING video for ${VID}"
    continue
  fi
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" -i "$SRC" \
    -frames:v 1 "$OUT/${VID}_t${T}.png" && echo "[triage] ok ${VID} t=${T}"
done
ls -la "$OUT"
