#!/bin/bash
# scan_game_screens_c96 完了待ち (最大 60 分ポーリング)
LOG=<home>/.gemini/antigravity/scratch/puyo_analyzer/logs/scan_c96_2026-08-08.log
for i in $(seq 1 360); do
  if ! pgrep -f scan_game_screens_c96 > /dev/null; then
    echo "[completed after ~$((i*10))s poll]"
    break
  fi
  sleep 10
done
echo "--- final tail ---"
tail -20 "$LOG"
echo "--- tsv line count ---"
wc -l <home>/.gemini/antigravity/scratch/puyo_analyzer/data/verify/c96_split_2026-08-08/scan_is_game.tsv
