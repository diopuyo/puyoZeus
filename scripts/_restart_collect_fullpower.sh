#!/bin/bash
# WSL再起動後の全力再開スクリプト (2026-07-20)。
#   1. 完了済み窓 (log に "rows" あり) をスキップして残ジョブを生成
#   2. 必要な動画を ext4 ($HOME/frames) にコピー (/mnt/c の 9p I/O ボトルネック回避)
#   3. _run_safe.sh MAXPAR=12 で detach 起動
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

JOBS=scripts/_jobs_collect_xii_rest.txt
: > "$JOBS"
mkdir -p "$HOME/frames"

for n in 29 30 31 32 33 34 35 36 37 38; do
  need_video=0
  for suf in "" "_gap" "_mid"; do
    log="logs/collect_xii_v${n}${suf}.log"
    if ! grep -q "rows" "$log" 2>/dev/null; then
      need_video=1
      case "$suf" in
        "")     args="--max-sec 300" ;;
        "_gap") args="--start-sec 300 --max-sec 600" ;;
        "_mid") args="--start-sec 1200 --max-sec 360" ;;
      esac
      echo "./venv/bin/python -m scripts._collect_1t --video $HOME/frames/video_${n}.mp4 --out data/indicators_v2/study/v${n}${suf}.csv $args --board-npz data/indicators_v2/boards/v${n}${suf}.npz > logs/collect_xii_v${n}${suf}.log 2>&1"
    fi
  done >> "$JOBS"
  if [ "$need_video" = "1" ] && [ ! -f "$HOME/frames/video_${n}.mp4" ]; then
    echo "[copy] video_${n}.mp4 -> ext4"
    cp "data/frames/video_${n}.mp4" "$HOME/frames/"
  fi
done

echo "[restart] 残ジョブ数: $(wc -l < "$JOBS")"
free -g | head -2
setsid -f bash -c "bash scripts/_run_safe.sh $JOBS 14 5 1 > logs/collect_xii_batch6_2026-07-20.log 2>&1 < /dev/null"
sleep 3
tail -2 logs/collect_xii_batch6_2026-07-20.log
echo "[restart] LAUNCHED"
