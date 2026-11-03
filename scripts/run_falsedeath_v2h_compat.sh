#!/bin/bash
# E35‴の3短区間を同じ条件で再生して互換性を確認する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
for mode in v2f v2g; do
for source in c5 mia8KCjr52g review; do
  bash scripts/run_falsedeath_v2h.sh "compat_${mode}_$source" -m scripts.falsedeath_v2h_compat "$source" --mode "$mode"
done
done
