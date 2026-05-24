#!/usr/bin/env bash
# v89 (= あん vs ぷにちゃん, マスター・3 ブロック, video_id H9uHCNGqBqk)
# を DL → 75s クリップ → labeling 準備.
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

YTDLP="venv/bin/yt-dlp"
FFMPEG="venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
RAW_OUT="data/test_unknown/v89_match1_raw"
CLIP_OUT="data/test_unknown/v89_match1_75s_720p.mp4"

mkdir -p data/test_unknown logs

echo "[step1] yt-dlp 全 DL"
"$YTDLP" \
    --ffmpeg-location "$FFMPEG" \
    --concurrent-fragments 4 \
    -f "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]" \
    --merge-output-format mp4 \
    -o "${RAW_OUT}.%(ext)s" \
    "https://www.youtube.com/watch?v=H9uHCNGqBqk"

DL_FILE=$(ls -1 "${RAW_OUT}".mp4 "${RAW_OUT}".mkv "${RAW_OUT}".webm 2>/dev/null | head -1)
if [ -z "$DL_FILE" ]; then
    echo "[error] DL 失敗" >&2
    exit 1
fi

echo "[step2] 195-270s クリップ整形"
"$FFMPEG" -y -ss 195 -i "$DL_FILE" -t 75 -c copy "$CLIP_OUT"
rm -f "$DL_FILE"

echo "[done] $CLIP_OUT"
