#!/usr/bin/env bash
# 得点タイブレーク無効化 ON/OFF の比較フレーム。
# t=29  : 1P 優位の盤面が 2P有利80% と表示されていた場面
# t=45  : 中盤
# t=54.5/58/62 : 同時連鎖中 (「どんどん2P有利へ寄る」症状の帯)
# t=66  : 1P 撃ち切り・2P 窒息寸前
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
DEMO=data/verify/youtube_demo_2026-08-07
OUT="$DEMO/no_score_lead_check"
mkdir -p "$OUT"
FF=$("$ROOT/venv/bin/python" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
for T in 29.0 45.0 54.5 58.0 62.0 66.0; do
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" \
    -i "$DEMO/demo_final_A_advantage.mp4" -frames:v 1 "$OUT/with_bias_t${T}.png"
  "$FF" -nostdin -hide_banner -loglevel error -y -ss "$T" \
    -i "$DEMO/_ab_no_score_lead.mp4" -frames:v 1 "$OUT/no_bias_t${T}.png"
done
ls "$OUT"
