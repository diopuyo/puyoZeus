#!/bin/bash
# 幕間フラグ (Q-01 本体) を含む 3 要素構成の 8 動画 A/B (2026-08-24、user 判断=案B)。
#
# 比較対象は data/verify/w37_rebaseline_2026-08-24/ の
#   off.json (掛け算式なし) / on.json (式読取 2 要素のみ)
# 本ドライバはそこへ enable_formula_step_interlude を足した構成を測る。
# W36/W37 是正後の物差しなので、3 者を同じ条件で比べられる。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
OUT=data/verify/w37_rebaseline_2026-08-24
ARGS="--videos v29,v40,v51,v57,v70,v89,v95,v97 --holdout v89,v97 --workers 4 --game-event-chain-exit"

PYTHONPATH=. nice -n 5 ./venv/bin/python scripts/measure_stable_cell_acc.py $ARGS \
  --enable-formula-freeze-fix --enable-formula-step-interlude \
  --output "$OUT/on_interlude.json" > "$OUT/on_interlude.log" 2>&1
echo done_on_interlude >> "$OUT/progress.log"
