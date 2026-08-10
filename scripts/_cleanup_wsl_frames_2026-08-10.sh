#!/bin/bash
# WSL側 ~/frames の収集済み動画コピーを削除する (2026-08-10 ストレージ緊急対応)
# 温存: _hold_video_c96.mp4 (デモ元・保持マーク付き) / video_c109.mp4 (走行中ジョブ使用中) / npzなし
set -u
IDS=/mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/612f6848-c9c6-432e-bd87-c4e4b11c6edf/scratchpad/processed_ids.txt
cd "$HOME/frames" || exit 1
count=0
freed=0
while read -r vid; do
  f="video_${vid}.mp4"
  if [ -f "$f" ] && [ "$f" != "video_c109.mp4" ]; then
    sz=$(stat -c%s "$f")
    rm -f "$f"
    freed=$((freed + sz))
    count=$((count + 1))
  fi
done < "$IDS"
echo "deleted=$count freed_GB=$((freed / 1073741824))"
du -sh "$HOME/frames"
