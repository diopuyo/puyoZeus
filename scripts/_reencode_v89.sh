#!/usr/bin/env bash
# v89 動画 AV1 → H.264 再エンコード (= opencv の AV1 decode 非対応対策).
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

FFMPEG="venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
SRC="data/test_unknown/v89_match1_75s_720p.mp4"
TMP="data/test_unknown/v89_match1_75s_720p_h264.mp4"

if [ ! -f "$SRC" ]; then
    echo "[error] source not found: $SRC" >&2
    exit 1
fi

echo "[step1] AV1 → H.264 再エンコード中..."
"$FFMPEG" -y -i "$SRC" -c:v libx264 -preset fast -crf 23 -c:a aac "$TMP"

if [ ! -f "$TMP" ]; then
    echo "[error] re-encode failed, $TMP not created" >&2
    exit 1
fi

echo "[step2] 旧 AV1 を H.264 で上書き"
mv "$TMP" "$SRC"

echo "[step3] codec 確認"
"$FFMPEG" -i "$SRC" 2>&1 | grep -E "Video:|Audio:" | head -3

echo "[done] $SRC ready for labeling"
