#!/bin/bash
# 補正側 seed0 は G_fe が BLAS スレッド差で v5_common と一致しない (予測差最大 .014) ため再利用せず、rejudge の T_rs0 で再生する。
R=/mnt/d/puyo_analyzer/wt_nextfix/logs/next_shift_rejudge
T=/mnt/d/puyo_analyzer/wt_nextfix/logs/next_shift_train
until [ -f $R/all.done ]; do sleep 20; done
pids=""
for s in q_7gc4TgFig fcXG83vInDY mia8KCjr52g review zenchi; do bash $T/replay.sh J_T_rs0 $s & pids="$pids $!"; done
wait $pids
bash $T/score.sh J_T_rs0
echo done > $R/t0.done
