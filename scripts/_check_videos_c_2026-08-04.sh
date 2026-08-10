#!/bin/bash
# 効果測定(c) 対象31動画の現存確認 (2026-08-04、使い捨て)
missing=0
for v in c10 c11 c12 c13 c15 c16 c17 c18 c19 c20 c21 c22 c23 c24 c25 c26 c28 c29 c31 c34 c35 c36 c37 c41 c42 c44 c5 c68 c73 c75 c81; do
  if [ ! -f "/home/ryouj/frames/video_${v}.mp4" ]; then
    echo "MISSING: ${v}"
    missing=$((missing+1))
  fi
done
echo "CHECK_DONE missing=${missing}"
