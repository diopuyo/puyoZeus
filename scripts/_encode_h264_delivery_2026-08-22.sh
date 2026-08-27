#!/bin/bash
# 納品物を h264 に変換して軽くする (2026-08-22、user 指示)。
#
# ## なぜ必要か
#
# OpenCV の VideoWriter は **mpeg4** で書き出す (約9.9Mbps)。
# 1時間で約4.3GB、2本で8.4GB と大きく、圧縮効率が悪い。
#
# ## 実測 (120秒サンプルで検証、2026-08-22)
#
# | 設定 | ビットレート | 全編推定 | 画質 |
# |---|---|---|---|
# | 現状 (mpeg4) | 9.9Mbps | 8.4GB | — |
# | h264 crf20 | 3.1Mbps | **2.7GB** | SSIM 0.9955 / PSNR 46.5dB |
# | h264 crf23 | 2.1Mbps | 1.9GB | SSIM 0.9936 / PSNR 44.5dB |
#
# crf20 を採る (視覚的にほぼ無劣化、サイズは1/3)。所要は全編18〜20分の見積もり。
#
# **音声は再エンコードしない** (`-c:a copy`)。既に aac 192k で載っているため。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

OUT=data/verify/zenchi_delivery_2026-08-21
CRF=20

FFMPEG=$(PYTHONPATH=. ./venv/bin/python -c \
  "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)
if [ -z "$FFMPEG" ] || [ ! -x "$FFMPEG" ]; then
  echo "=== 中止: ffmpeg が見つからない ==="
  exit 1
fi

count_frames() {
  "$FFMPEG" -v error -stats -i "$1" -map 0:v:0 -f null - 2>&1 |
    tr '\r' '\n' | grep -oE "frame= *[0-9]+" | tail -1 | grep -oE "[0-9]+"
}

echo "=== h264 変換 start $(date +%F_%T) (crf=$CRF) ==="
cat /proc/loadavg

for n in 1 2; do
  SRC="$OUT/zenchi_set${n}_audio.mp4"
  DST="$OUT/zenchi_set${n}_h264.mp4"
  if [ ! -f "$SRC" ]; then
    echo "  set$n: **入力が無い** ($SRC)。結合が終わっていない可能性"
    continue
  fi
  BEFORE=$(du -m "$SRC" | cut -f1)
  NB_BEFORE=$(count_frames "$SRC")
  echo "--- set$n 変換開始 $(date +%H:%M:%S) (元 ${BEFORE}MB / ${NB_BEFORE}frames) ---"
  "$FFMPEG" -y -v error -stats -i "$SRC" \
    -c:v libx264 -crf "$CRF" -preset medium -pix_fmt yuv420p \
    -c:a copy "$DST" 2>&1 | tr '\r' '\n' | tail -2
  RC=$?
  AFTER=$(du -m "$DST" 2>/dev/null | cut -f1)
  NB_AFTER=$(count_frames "$DST")
  echo "  rc=$RC  ${BEFORE}MB -> ${AFTER}MB  ($(echo "scale=1; $AFTER * 100 / $BEFORE" | bc)%)"
  # フレーム数が変わっていないこと (欠落・重複ゼロの確認)
  if [ "$NB_BEFORE" = "$NB_AFTER" ]; then
    echo "  フレーム数 $NB_BEFORE -> $NB_AFTER **一致**"
  else
    echo "  **フレーム数が変わった** $NB_BEFORE -> $NB_AFTER (欠落または重複)"
  fi
done

echo "=== 完了 $(date +%F_%T) ==="
ls -l "$OUT"/*.mp4
cat /proc/loadavg
