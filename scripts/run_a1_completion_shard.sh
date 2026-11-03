#!/bin/bash
# 最終候補を重複のない二群に分け、同じ資源制限で再生する。
set -eu
cd /mnt/d/puyo_analyzer/wt_a1
export A1_MODES=a1
case "$1" in
  first) export A1_NAMES='review q_7gc4TgFig fcXG83vInDY mia8KCjr52g zenchi c1 c2 c3';;
  last) export A1_NAMES='c4 c5 c6 s0 s1 s2 s3 s4 s5 s6';;
  *) exit 2;;
esac
bash scripts/run_a1_completion.sh
