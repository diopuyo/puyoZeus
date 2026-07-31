#!/bin/bash
# 修正版レンダの検証フレーム抽出(v35ゲーム開始付近=幻の差が消えたか)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FF=$(./venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')
OUT="/mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/f3efc5f5-b2ab-4019-b80c-3a2d35f86017/scratchpad"
# v35: 動画41秒付近(ゲーム開始・空盤面)
"$FF" -y -ss 41 -i "data/indicators_v2/overlay/match_v35_g11_2Pwin_h264.mp4" -vframes 1 "$OUT/final_v35_41s.png" 2>/dev/null
ls -la "$OUT/final_v35_41s.png"
