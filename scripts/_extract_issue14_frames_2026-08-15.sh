#!/bin/bash
# 指摘14 証拠フレーム抽出 (final3_m1.mp4、デモ内相対秒 = 絶対秒-162)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=./venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
OUT=data/verify/diag_issue14_2026-08-15
mkdir -p "$OUT"
IN=data/verify/demo_fixed_2026-08-13/final3_m1.mp4

extract() {
  local t="$1"
  local label="$2"
  "$FF" -y -ss "$t" -i "$IN" -frames:v 1 "$OUT/frame_${label}_t${t}.png" -loglevel error
}

# t=絶対秒-162 (final3_m1 = 162-218s)
extract 32.53 pre_resolve   # 絶対194.53直前、まだ通常評価
extract 32.60 resolve_96pct # 絶対194.53、_resolve発火直後 hold=96.1%
extract 33.33 reeval_19pct  # 絶対195.33、_reevaluate_live_defenderで81.1%/18.9%へ
extract 36.53 hold_19pct_mid # 絶対199.03、18.9%凍結継続中
extract 38.53 hold_19pct_end # 絶対200.53、凍結終盤 (直後200.83に24.17%へ再評価)
extract 39.53 partial_resolve # 絶対201.53、score2確定後の再決着直後
extract 45.30 crash_to_death # 絶対207.30、pend1=216後2Pへ反転・0.7%へ向かう途中
echo DONE_EXTRACT
ls -la "$OUT"
