#!/bin/bash
# 勝者一致率の横展開検証を3並列で回すジョブプール(コード変更ゼロの回帰確認)。
# 各動画は自分自身を学習から除外(リーク防止)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
pkill -f "scripts.validate_advantage_winner" 2>/dev/null  # 前回の残骸掃除
sleep 2
MAX=3
for V in 31 32 33 34 35 36 37 38; do
  while [ "$(jobs -r | wc -l)" -ge "$MAX" ]; do sleep 5; done
  nice -n 10 ./venv/bin/python -u -m scripts.validate_advantage_winner \
    --video "data/frames/video_${V}.mp4" --video-id "video_${V}" \
    --start-sec 140 --end-sec 700 --exclude-video "video_${V}" \
    > "logs/validate_kill_v${V}.log" 2>&1 &
  echo "started video_${V} (running=$(jobs -r | wc -l))"
done
wait
echo "ALL VALIDATION DONE"
