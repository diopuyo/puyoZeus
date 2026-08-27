#!/bin/bash
# 分割点確認用: 候補6点(#1,2,3,5,6,7)の前後フレームを元動画から切り出し
# #5,6,7 はセット境界候補として広めの窓も取る
set -e
cd "/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
SRC="data/frames/video_zenchi_c0BQoMJwwQU.mp4"
OUTDIR="data/verify/zenchi_boundary_confirm_2026-08-21"
mkdir -p "$OUTDIR"

extract() {
  # $1 = 目標秒(実秒), $2 = 出力ファイル名
  t="$1"; name="$2"
  # 2段seek: 目標-3秒に高速seek → そこから3秒分デコードして正確に着地
  pre=$(awk -v t="$t" 'BEGIN{v=t-3; if(v<0)v=0; printf "%.3f", v}')
  off=$(awk -v t="$t" -v pre="$pre" 'BEGIN{printf "%.3f", t-pre}')
  ffmpeg -y -ss "$pre" -i "$SRC" -ss "$off" -frames:v 1 -q:v 2 "$OUTDIR/$name" >/dev/null 2>&1
}

# 近接窓 (-10s ~ +10s, 1秒刻み) : 全候補共通
for cand in "893.7:C1" "1738.3:C2" "2637.3:C3" "4379.5:C5" "5255.6:C6" "6131.6:C7"; do
  base="${cand%%:*}"; tag="${cand##*:}"
  for d in -10 -8 -6 -4 -2 -1 0 1 2 3 4 5 6 8 10; do
    t=$(awk -v b="$base" -v d="$d" 'BEGIN{printf "%.1f", b+d}')
    name=$(printf "%s_t%+03d.png" "$tag" "$d")
    extract "$t" "$name"
  done
done

echo "近接窓 done: $(ls "$OUTDIR" | wc -l) files"

# セット境界候補用の広域窓 (#5,6,7): -20s~+60s を4秒刻み (リザルト演出の長さを見る)
for cand in "4379.5:C5wide" "5255.6:C6wide" "6131.6:C7wide"; do
  base="${cand%%:*}"; tag="${cand##*:}"
  for d in $(seq -20 4 60); do
    t=$(awk -v b="$base" -v d="$d" 'BEGIN{printf "%.1f", b+d}')
    name=$(printf "%s_t%+04d.png" "$tag" "$d")
    extract "$t" "$name"
  done
done

echo "広域窓 done: $(ls "$OUTDIR" | wc -l) files"
