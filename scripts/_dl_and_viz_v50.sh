#!/usr/bin/env bash
# v50 (= 3 vs いさな、 マスター・2 ブロック、 video_id enHxRrKeAAs) を 195-270s で部分 DL し、
# 認識 → viz + diag 生成. cycle 71c (= 案 C 変種ガード入り) で評価.
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

YTDLP="venv/bin/yt-dlp"
FFMPEG="venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
RAW_OUT="data/test_unknown/v50_match1_raw"
CLIP_OUT="data/test_unknown/v50_match1_75s_720p.mp4"
VIZ_OUT="data/test_unknown/v50_match1_75s_viz_phase1c.mp4"
DIAG_OUT="data/diagnostics/v50_match1_75s_diag_phase1c.jsonl"

mkdir -p data/test_unknown data/diagnostics logs

echo "[step1] yt-dlp 全 DL (= partial DL ffmpeg segfault 回避)"
"$YTDLP" \
    --ffmpeg-location "$FFMPEG" \
    --concurrent-fragments 4 \
    -f "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]" \
    --merge-output-format mp4 \
    -o "${RAW_OUT}.%(ext)s" \
    "https://www.youtube.com/watch?v=enHxRrKeAAs"

# DL したファイルを 195-270s = 75s クリップに整形 (= ffmpeg seek + copy)
DL_FILE=$(ls -1 "${RAW_OUT}".mp4 "${RAW_OUT}".mkv "${RAW_OUT}".webm 2>/dev/null | head -1)
if [ -z "$DL_FILE" ]; then
    echo "[error] DL 失敗、 出力ファイル無し" >&2
    exit 1
fi
echo "[step2] ffmpeg で 195-270s クリップ整形: $DL_FILE -> $CLIP_OUT"
"$FFMPEG" -y -ss 195 -i "$DL_FILE" -t 75 -c copy "$CLIP_OUT"
echo "[step2b] 元動画削除 (= ストレージ管理ルール)"
rm -f "$DL_FILE"

echo "[step3] visualize_recognition (phase1c)"
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$CLIP_OUT" \
    --output "$VIZ_OUT" \
    --max-sec 75

echo "[step4] diagnose_chain_transitions (phase1c)"
PYTHONPATH=. ./venv/bin/python -m scripts.diagnose_chain_transitions \
    --video "$CLIP_OUT" \
    --output "$DIAG_OUT" \
    --max-sec 75 \
    --progress-every 500

echo "[step5] analyze diag"
PYTHONPATH=. ./venv/bin/python -m scripts.analyze_chain_diag \
    --input "$DIAG_OUT" \
    --output "${DIAG_OUT%.jsonl}_summary.md"

echo "[done] viz=$VIZ_OUT diag=$DIAG_OUT"
