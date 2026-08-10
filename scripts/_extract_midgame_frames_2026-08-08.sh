#!/usr/bin/env bash
# user 指摘「29秒は 1P のほうが色ぷよの量が多く有利なはず、 連鎖も 1P のほうが
# 組めている」の確認用。 中盤の推移も併せて抜く。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
DEMO=data/verify/youtube_demo_2026-08-07
OUT="$DEMO/midgame_check"
mkdir -p "$OUT"
FF=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
for T in 29.0 32.0 36.0 40.0 46.0 50.0; do
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" \
    -i "$DEMO/demo_A_advantage_only_efire_2026-08-08.mp4" -frames:v 1 "$OUT/efire_t${T}.png"
done
ls "$OUT"
