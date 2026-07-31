#!/bin/bash
# #43: c系93本の勝敗抽出(WIN★パネル差分方式、コミット99a7172)を夜間一括実行。
# 22:00以降開始(21-22時はCPU/メモリ25%空けのuser指示)。3並列・nice。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
OUT=data/verify/winners_panel_diff_2026-07-26
LOG=logs/winners_panel_diff_all93_2026-07-26.log
mkdir -p "$OUT"

# 22:05まで待機(静音時間明け)
now=$(date +%s); start=$(date -d "22:05" +%s)
if [ "$now" -lt "$start" ]; then sleep $((start - now)); fi
echo "[all93] start $(date)" >> "$LOG"

run_one() {
  v="$1"
  stem=$(basename "$v" .mp4)
  out="$OUT/${stem}.json"
  if [ -s "$out" ]; then echo "[skip] $stem" >> "$LOG"; return; fi
  PYTHONPATH=. nice -n 10 ./venv/bin/python -m scripts.extract_match_winners \
    --video "$v" --out-json "$out" --panel-diff-mode >> "$LOG" 2>&1
  echo "[done] $stem $(date '+%H:%M')" >> "$LOG"
}

N=0
for v in data/frames/video_c*.mp4; do
  run_one "$v" &
  N=$((N+1))
  if [ $((N % 3)) -eq 0 ]; then wait; fi
done
wait
echo "[all93] ALL DONE $(date)" >> "$LOG"
