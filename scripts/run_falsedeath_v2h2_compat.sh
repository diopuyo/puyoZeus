#!/bin/bash
# 既存v2hの条件を変えず、今回の既定OFFが原票と一致することを検証する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
for source in c5 mia8KCjr52g review; do
  bash scripts/run_falsedeath_v2h2.sh "compat_v2h_$source" -m scripts.falsedeath_v2h2_compat "$source" --mode v2h
done
