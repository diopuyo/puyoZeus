#!/usr/bin/env bash
# 最終デモ 4 本の検収用フレーム抽出。
# t=29 (user 指摘: 1P の色ぷよが多く有利なはず)、 t=54.5 / 58 (大連鎖中の逆転)、
# t=66 (1P 撃ち切り・2P 窒息寸前) を確認する。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
DEMO=data/verify/youtube_demo_2026-08-07
OUT="$DEMO/final_check"
mkdir -p "$OUT"
FF=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
for T in 29.0 54.5 58.0 66.0; do
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" \
    -i "$DEMO/demo_final_A_advantage.mp4" -frames:v 1 "$OUT/A_t${T}.png"
done
for T in 26.2 57.6; do
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" \
    -i "$DEMO/demo_final_C_recognition.mp4" -frames:v 1 "$OUT/C_t${T}.png"
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" \
    -i "$DEMO/demo_final_D_no_cell_overlay.mp4" -frames:v 1 "$OUT/D_t${T}.png"
done
ls "$OUT"
