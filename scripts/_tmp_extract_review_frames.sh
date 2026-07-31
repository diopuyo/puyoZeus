#!/bin/bash
# レビュー区間(動画45-62秒)のフレームを2秒刻みで抽出して現象を確認
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
IN="data/indicators_v2/overlay/match_v30_g18_1Pwin_h264.mp4"
OUT="/mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/f3efc5f5-b2ab-4019-b80c-3a2d35f86017/scratchpad"
mkdir -p "$OUT"
for t in 46 48 50 52 54 56 58 60; do
  "$FF" -y -ss "$t" -i "$IN" -vframes 1 "$OUT/rev_${t}s.png" 2>/dev/null
done
ls -la "$OUT"/rev_*.png | awk '{print $5, $9}'
