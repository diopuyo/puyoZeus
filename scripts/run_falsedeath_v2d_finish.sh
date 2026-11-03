#!/bin/bash
# 完了した全体再生を採点した後に、全表と受領票を新規保存する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2d
bash scripts/run_falsedeath_v2d.sh static_check -m scripts.falsedeath_v2d_static_checks
bash scripts/run_falsedeath_v2d.sh history -m scripts.falsedeath_v2d_history
if /usr/bin/python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$OUT/LADDER.json"; then
  bash scripts/run_falsedeath_v2d.sh omissions -m scripts.falsedeath_v2d_omissions
  bash scripts/run_falsedeath_v2d.sh supplement -m scripts.falsedeath_v2d_supplement
  bash scripts/run_falsedeath_v2d.sh verify -m scripts.falsedeath_v2d_verify
fi
bash scripts/run_falsedeath_v2d.sh receipt -m scripts.falsedeath_v2d_receipt
bash scripts/run_falsedeath_v2d.sh report -m scripts.falsedeath_v2d_report
