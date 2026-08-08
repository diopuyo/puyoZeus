#!/usr/bin/env bash
# 試合終盤の有利不利表示を確認する (user 指摘「最後の連鎖を撃った瞬間に
# 勝利が確定しているのに、 モデルがそれを確定として扱えていないのでは」)。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
DEMO=data/verify/youtube_demo_2026-08-07
OUT="$DEMO/endgame_check"
mkdir -p "$OUT"
FF=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
for T in 62.0 64.0 66.0 68.0 70.0 74.0 78.0; do
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" \
    -i "$DEMO/demo_A_advantage_only_efire_2026-08-08.mp4" -frames:v 1 "$OUT/efire_t${T}.png"
done
ls "$OUT"
