#!/bin/bash
# olRyxDGacbg (A級 DIO vs TS) の5試合 (game_idx 1-5、t=2887-3231s付近) に
# 有利不利判定オーバーレイ (対称化+タイムライン密度 修正版) を合成する。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
OUT_DIR="data/verify/advantage_videos_olRyxDGacbg_2026-08-03"
mkdir -p "$OUT_DIR"
DELTA_CSV="data/verify/delta_winprob_olRyxDGacbg_2026-08-03/exchange_delta_winprob.csv"
NPZ_DIR="data/indicators_v2/boards_lean_olRyxDGacbg_2026-08-03"
FF="$(pwd)/venv/bin"

idx=1
for game in 1 2 3 4 5; do
  mnum=$(printf "%02d" "$idx")
  echo "[match_${mnum}] game_idx=${game} レンダ開始 $(date)"
  PYTHONPATH=. ./venv/bin/python -m scripts.render_delta_winprob_demo \
    --video-id olRyxDGacbg --game-idx "$game" \
    --npz-dir "$NPZ_DIR" --delta-winprob-csv "$DELTA_CSV" \
    --out-dir "$OUT_DIR"
  raw="${OUT_DIR}/delta_winprob_demo_olRyxDGacbg_g${game}.mp4"
  final="${OUT_DIR}/match_${mnum}.mp4"
  if [ -s "$raw" ]; then
    "${FF}/ffmpeg" -y -i "$raw" -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p \
      -c:a aac -movflags +faststart "$final"
    rm -f "$raw"
    echo "[match_${mnum}] 完了 -> ${final} $(date)"
  else
    echo "[match_${mnum}] 失敗: ${raw} が生成されていません" >&2
  fi
  idx=$((idx+1))
done
echo "[all done] $(date)"
ls -la "$OUT_DIR"
