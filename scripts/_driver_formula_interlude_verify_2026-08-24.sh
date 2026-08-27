#!/bin/bash
# Q-01 修正後の偽イベント再測定ドライバ (2026-08-24)。
#
# 目的: Q-02 の唯一残ったブロッカー「重複候補 (FLICKER) の増加」が、
#       Q-01 (掛け算式の段が進まない) の症状だったのかを検証する。
#
# 先行測定 (scripts/_driver_formula_fix_verify_2026-08-24.sh の第4節) と
# **同じ窓・同じ動画・同じ OFF 構成**で流し、ON 構成にだけ
# enable_formula_step_interlude=True を足す。
# 出力先は別ディレクトリで、先行の成果物は一切上書きしない。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
PY="./venv/bin/python"
OUT=logs/_probe_formula_interlude_2026-08-24
mkdir -p "$OUT"

for win in "780 1080 w1" "3200 3500 w2"; do
  set -- $win
  for mode in off on; do
    PYTHONPATH=. nice -n 10 $PY scripts/_probe_formula_interlude_2026-08-24.py \
      "$mode" "$1" "$2" "$3" > "$OUT/probe_$3_${mode}.log" 2>&1
    echo "done $3 $mode" >> "$OUT/driver_progress.log"
  done
done
echo ALL_DONE >> "$OUT/driver_progress.log"
