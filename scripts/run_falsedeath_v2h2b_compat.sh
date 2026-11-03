#!/bin/bash
# D-v2h2の既存短区間を補正OFFで再生してバイト一致を確認する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
for source in c5 mia8KCjr52g review; do
  bash scripts/run_falsedeath_v2h2b.sh "compat_v2h2_$source" -m scripts.falsedeath_v2h2b_compat "$source" --mode v2h2
done
