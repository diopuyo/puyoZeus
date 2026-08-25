#!/bin/bash
# STABLE凍結デッドロック根治の検証ジョブ逐次ドライバ (2026-08-24 コーダ)。
# 1. E2E ON (別途起動済み) の完了を待つ
# 2. E2E OFF (陽性対照)
# 3. 大連鎖10ケース プローブ (フラグON)
# 4. 偽イベント率 before/after (2ウィンドウ × off/on)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
PY="./venv/bin/python"
export PYTHONPATH=.

# 1. E2E ON の完了待ち
while pgrep -f "_diag_formula_fix_e2e_2026-08-24.py on" > /dev/null; do
  sleep 30
done

# 2. E2E OFF (陽性対照)
nice -n 10 $PY scripts/_diag_formula_fix_e2e_2026-08-24.py off \
  > logs/_diag_formula_fix_e2e_2026-08-24_off.log 2>&1

# 3. 10ケース (フラグON)
OUT=logs/_probe_formula_fix_cases_2026-08-24
mkdir -p "$OUT"
while read -r case t0 t1; do
  nice -n 10 $PY scripts/_probe_formula_fix_cases_2026-08-24.py "$case" "$t0" "$t1" \
    > "$OUT/probe_${case}.log" 2>&1
  echo "done $case" >> "$OUT/driver_progress.log"
done <<CASES
c01_seg08_1P 6702.5 6718.9
c02_seg08_2P 6488.1 6510.4
c03_seg05_2P 4249.9 4262.4
c04_seg06_2P 5229.0 5250.1
c05_seg02_2P 1476.0 1495.9
c06_seg07_1P 5455.5 5477.6
c07_seg01_2P 874.3 885.6
c08_seg01_2P 792.7 808.7
c09_seg07_1P 5570.9 5585.9
c10_seg04_1P 3304.5 3323.6
CASES

# 4. 偽イベント率 (2ウィンドウ × off/on)
OUT2=logs/_probe_formula_false_event_2026-08-24
mkdir -p "$OUT2"
for win in "780 1080 w1" "3200 3500 w2"; do
  set -- $win
  for mode in off on; do
    nice -n 10 $PY scripts/_probe_formula_false_event_2026-08-24.py \
      "$mode" "$1" "$2" "$3" > "$OUT2/probe_$3_${mode}.log" 2>&1
    echo "done $3 $mode" >> "$OUT2/driver_progress.log"
  done
done
echo ALL_DONE >> "$OUT2/driver_progress.log"
