#!/bin/bash
# Claude 起動済み検証のクリティカルパスを優先する一時監視器。
# 既存ジョブを停止せず、後続で生成される子プロセスの nice 値だけを調整する。

FORMULA_PATTERN='[p]ython .*(_diag_formula_fix_e2e_2026-08-24.py|_probe_formula_fix_cases_2026-08-24.py|_probe_formula_false_event_2026-08-24.py|measure_stable_cell_acc.py)'
PM100_PATTERN='[p]ython -m scripts.visualize_advantage_overlay.*pm100_fix_2026-08-24/dumps_'
DRIVER_PATTERN='[b]ash scripts/_driver_formula_fix_(verify|backtest)_2026-08-24.sh'

prioritize_matches() {
  local priority="$1"
  local pattern="$2"
  local process_id

  while read -r process_id; do
    if [[ -n "$process_id" ]]; then
      renice -n "$priority" -p "$process_id" >/dev/null 2>&1 || true
    fi
  done < <(pgrep -f "$pattern" || true)
}

while pgrep -f "$DRIVER_PATTERN" >/dev/null; do
  prioritize_matches 0 "$FORMULA_PATTERN"
  prioritize_matches 10 "$PM100_PATTERN"
  sleep 30
done
