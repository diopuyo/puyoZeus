#!/bin/bash
# 評価セットの収集 (p3check: 測定器の検査、p1/p2: 第1〜40試合)。3並列 (CYCLE_FINDINGS の収集上限)。
# 起動: MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/launch_collect.sh
L=/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set
for part in p3check p1 p2; do
  setsid -f bash -c "nice -n 10 bash $L/run_in.sh -m scripts.collect_eval_set_20261001 --part $part > $L/collect_$part.log 2>&1; echo \$? > $L/collect_$part.done" < /dev/null > /dev/null 2>&1
done
