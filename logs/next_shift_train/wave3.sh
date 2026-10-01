#!/bin/bash
# 診断の再生 (再学習ゆらぎ): wave2 の完了を待ってから S_orig_rs1 → S_T_rs1
W=/mnt/d/puyo_analyzer/wt_nextfix/logs/next_shift_train
until [ -f $W/wave2.done ]; do sleep 10; done
for v in S_orig_rs1 S_T_rs1; do
  for s in q_7gc4TgFig fcXG83vInDY mia8KCjr52g review zenchi; do
    bash $W/replay.sh $v $s &
  done
  wait
  bash $W/score.sh $v
done
echo done > $W/wave3.done
