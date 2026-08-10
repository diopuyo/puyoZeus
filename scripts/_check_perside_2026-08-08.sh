#!/usr/bin/env bash
# 片側独立更新 ON/OFF の比較フレーム。
# t=29  : user 指摘「1P の色ぷよが多く有利なはず」
# t=54.5: 1P 9連鎖の発火直後
# t=66  : 1P 撃ち切り・2P 窒息寸前 (勝利ほぼ確定)
# t=62/70: 62%→50% のノコギリ波が見えた帯
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
DEMO=data/verify/youtube_demo_2026-08-07
OUT="$DEMO/perside_check"
mkdir -p "$OUT"
FF=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
for T in 29.0 54.5 62.0 66.0 70.0; do
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" \
    -i "$DEMO/demo_final_A_advantage.mp4" -frames:v 1 "$OUT/off_t${T}.png"
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" \
    -i "$DEMO/_ab_perside_on.mp4" -frames:v 1 "$OUT/on_t${T}.png"
done
ls "$OUT"
