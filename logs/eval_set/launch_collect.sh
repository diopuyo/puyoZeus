#!/bin/bash
# 評価セットの収集。引数の区間名を並列で起動する (例: c1 c2 c3 c4 c5)。
# 起動: MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/launch_collect.sh c1 c2 ...
# 初回 (12:23) は p3check p1 p2 で起動。p1/p2 は CPU 競合で約5時間かかる見込みのため 12:59 に停止し c1〜c6 に切替。
L=/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set
for part in "$@"; do
  setsid -f bash -c "nice -n 10 bash $L/run_in.sh -m scripts.collect_eval_set_20261001 --part $part > $L/collect_$part.log 2>&1; echo \$? > $L/collect_$part.done" < /dev/null > /dev/null 2>&1
done
