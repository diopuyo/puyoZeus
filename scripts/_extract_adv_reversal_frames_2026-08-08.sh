#!/usr/bin/env bash
# 大連鎖前後の有利不利表示を、 早期発火反応の有無で比較する。
# 1P は t=53.97 に 9 連鎖、 2P は t=56.47 に 7 連鎖を撃つ。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
DEMO=data/verify/youtube_demo_2026-08-07
OUT="$DEMO/adv_reversal_check"
mkdir -p "$OUT"
FF=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
for T in 52.0 54.5 56.0 58.0 60.0 62.0; do
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" \
    -i "$DEMO/demo_A_advantage_only_2026-08-08.mp4" -frames:v 1 "$OUT/base_t${T}.png"
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" \
    -i "$DEMO/demo_A_advantage_only_efire_2026-08-08.mp4" -frames:v 1 "$OUT/efire_t${T}.png"
done
ls "$OUT"
