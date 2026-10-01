#!/bin/bash
# 使い方: bash replay_all.sh <variant>   5記録を並列に detach 起動する
for s in q_7gc4TgFig fcXG83vInDY mia8KCjr52g review zenchi; do
  setsid -f bash /mnt/d/puyo_analyzer/wt_nextfix/logs/next_shift_train/replay.sh "$1" "$s"
done
