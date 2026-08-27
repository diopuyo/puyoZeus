#!/bin/bash
# 条件5の会計検収とOFF表示比較を同じ成果物集合から実行する。
set -euo pipefail
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

OFF=data/verify/gate4_formal_dense_2026-08-26/cond1_off_baseline
ON=data/verify/gate4_condition5_2026-08-26/cond5_exchange_episode_v12
LOG=logs/gate4_condition5_2026-08-26
TRUTH=data/verify/gate4_formal_dense_2026-08-26/win_panel_truth.tsv
mkdir -p "$LOG"
LEDGER_LOG="$LOG/condition5_v12_ledger_verify.log"
DISPLAY_LOG="$LOG/condition5_v12_display_vs_off.log"
for path in "$LEDGER_LOG" "$DISPLAY_LOG"; do
  if [[ -e "$path" ]]; then
    echo "既存検収logは上書きしない: $path" >&2
    exit 1
  fi
done

./venv/bin/python scripts/_verify_gate4_condition5_2026-08-26.py "$ON" \
  | tee "$LEDGER_LOG"
./venv/bin/python scripts/_analyze_pm100_display_pair_2026-08-26.py \
  "$OFF" "$ON" "$TRUTH" \
  | tee "$DISPLAY_LOG"
