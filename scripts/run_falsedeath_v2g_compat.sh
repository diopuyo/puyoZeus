#!/bin/bash
# E35‴の3短区間を同じ条件で再生して互換性を確認する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
for source in c5 mia8KCjr52g review; do
  bash scripts/run_falsedeath_v2g.sh "compat_$source" -m scripts.falsedeath_v2g_compat "$source"
done
