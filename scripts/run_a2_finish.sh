#!/bin/bash
# 再生と独立参照の完了後にだけ最終検収する。
set -eu
cd /mnt/d/puyo_analyzer/wt_a2
while [ ! -f logs/a2_safety/RESULT.json ] || [ ! -f logs/a2_safety/replay/reference/zenchi/status.json ]; do
  sleep 20
done
bash scripts/run_a2_safety.sh verification -m scripts.a2_verify
bash scripts/run_a2_safety.sh start_audit -m scripts.a2_start_audit
bash scripts/run_a2_safety.sh case6_final -m scripts.a2_case6
bash scripts/run_a2_safety.sh report_final -m scripts.a2_report
