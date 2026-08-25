#!/bin/bash
# 新2列 (match_progress / ojama_damage_forecast) の効果測定 (2026-08-21)。
#
# 62本の npz から学習CSVを作り直して再学習する。比較対象は同じ62本で
# 2列なしの結果 (data/verify/retrain_model62_2026-08-21、AUC 0.6354 /
# 中盤 0.5497)。動画も収集構成も同一なので、差は追加した2列だけに帰属する。
#
# 失敗基準 (fable 設計、先に固定しておく):
#   中盤 pooled AUC +0.005 未満 かつ 新列の permutation が ≤2σ なら不採用。
#   これを満たさないのに「効いた」と報告しない (測定器事故12件の教訓)。
#
# 注意: match_progress は side 対称 (両者同値) なので単独では勝敗を識別
# できない。単独 permutation が小さくても「効いていない」とは判定できず、
# 効果は交互作用経由でのみ現れる。判定は中盤 AUC の変化を主に見る。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
exec >> logs/relearn_3col_2026-08-21.log 2>&1

NPZ_DIR=data/indicators_v2/boards_lean_model50v2_2026-08-20
CSV_DIR=data/verify/labeled_win_model62_3col_2026-08-21
OUT_DIR=data/verify/retrain_model62_3col_2026-08-21

echo "=== 3列追加版の学習 (color_offset_power 追加) start $(date +%F_%T) ==="
N=$(ls "$NPZ_DIR"/*.npz 2>/dev/null | wc -l)
echo "[入力] npz $N 本"
if [ "$N" -lt 62 ]; then
  echo "[中止] 62本揃っていない ($N 本)"
  exit 1
fi

echo "--- CSVビルド $(date +%T) ---"
mkdir -p "$CSV_DIR"
PYTHONPATH=. ./venv/bin/python -m scripts.build_labeled_win_from_npz \
    --npz-dir "$NPZ_DIR" \
    --out "$CSV_DIR/labeled_win_model62_3col.csv" \
    --profile full \
    --exclude-match-end-locked
echo "  rc=$?"

echo "--- 再学習 $(date +%T) ---"
PYTHONPATH=. ./venv/bin/python scripts/_retrain148_2026-08-14.py \
    --csv "$CSV_DIR/labeled_win_model62_3col.csv" \
    --out-dir "$OUT_DIR"
echo "  rc=$?"

echo "=== 完了 $(date +%F_%T) ==="
