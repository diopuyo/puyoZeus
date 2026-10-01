#!/bin/bash
# 診断の再生: A_snew → A_cnew → R3 を順に (1波 5並列)、各波の終わりに採点する
W=/mnt/d/puyo_analyzer/wt_nextfix/logs/next_shift_train
for v in A_snew A_cnew R3; do
  for s in q_7gc4TgFig fcXG83vInDY mia8KCjr52g review zenchi; do
    bash $W/replay.sh $v $s &
  done
  wait
  bash $W/score.sh $v
done
echo done > $W/wave2.done
