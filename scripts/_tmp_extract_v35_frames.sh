#!/bin/bash
# v35レビュー区間(39-44秒)のフレーム抽出
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
IN="data/indicators_v2/overlay/match_v35_g11_2Pwin_h264.mp4"
OUT="/mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/f3efc5f5-b2ab-4019-b80c-3a2d35f86017/scratchpad"
for t in 39 40 41 42 43 44; do
  "$FF" -y -ss "$t" -i "$IN" -vframes 1 "$OUT/v35_${t}s.png" 2>/dev/null
done
ls "$OUT"/v35_*.png
