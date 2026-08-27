#!/bin/bash
# Gate 3R-6 P1 是正 (境界の正式受理ラッチ、2026-08-26) の再検収一式。
#
# Codex 第26報レビュー要件5・6:
#   5. 修正後、同一窓で再測し total_boundaries が実際の境界数と一致することを
#      母数付きで示す。
#   6. 真の窒息 / 既知誤判定 / まちうけ / ON・OFF / gross併用 / 境界確定3件を
#      再検収する。
#   7. 既存成果物を上書き・中断しない
#      -> `--out-suffix _p1fix` で別ファイル名にする。既存の
#         first5games_boundaryconfirm_*.npz には一切書かない。
#
# OFF は3run 取り直す。P1 是正は `resolve_boundary_confirmations` の呼出回数を
# 変えるが、`enable_death_confirm_sequence=False` の OFF では death 列が
# dump に出ないため bit-identical であるはず。「はず」で済ませず実測する。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
PY=./venv/bin/python

echo "=== Gate3R-6 P1是正 再検収 start $(date +%F_%T) ==="
for SPEC in "off:_p1v2_run1" "off:_p1v2_run2" "off:_p1v2_run3" \
            "on:_p1v2" "gross:_p1v2" "on_gross:_p1v2"; do
  MODE="${SPEC%%:*}"
  SUF="${SPEC##*:}"
  echo "--- mode=$MODE suffix=$SUF 開始 $(date +%F_%T) ---"
  nice -n 19 $PY -m scripts._diag_gate3r6_boundary_confirm_dump_2026-08-25 \
    --mode "$MODE" --out-suffix "$SUF"
  echo "--- mode=$MODE rc=$? $(date +%F_%T) ---"
done
echo "=== 再検収 全完了 $(date +%F_%T) ==="
echo "P1FIX_RECHECK_DONE"
