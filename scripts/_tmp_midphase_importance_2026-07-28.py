"""中盤(手数固定境界)限定の permutation importance を算出する (#43 段階2 m20用)。

## 背景
model_indicator_win.py の permutation importance (compute_perm_importance_win)
は全位相まとめて計算しており、中盤特有の重要指標が他位相の寄与で埋没する
可能性がある。本スクリプトは同一関数を中盤サブセットのみに適用し、
c20 の全位相 importance (board_ojama_count系/conn_triple/max_column_height)
と中盤限定版を比較する。

model_indicator_win.py は変更しない (import して関数を再利用するのみ、
既存互換完全維持)。read-only 分析用の使い捨てスクリプト。

## 使い方
    PYTHONPATH=. ./venv/bin/python -m scripts._tmp_midphase_importance_2026-07-28 \
        --labeled data/verify/labeled_win_m20_2026-07-28/labeled_win_m20.csv \
        --fixed-q33 18 --fixed-q67 40 \
        --out data/verify/win_eval_m20_2026-07-28/m20_midphase_importance.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.model_indicator_win as miw  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="中盤限定 permutation importance")
    parser.add_argument("--labeled", required=True, help="labeled_win.csv パス")
    parser.add_argument("--fixed-q33", type=float, required=True, help="序盤/中盤境界(手数)")
    parser.add_argument("--fixed-q67", type=float, required=True, help="中盤/終盤境界(手数)")
    parser.add_argument("--out", required=True, help="出力 CSV パス")
    args = parser.parse_args()

    print(f"[midphase_importance] labeled={args.labeled}")
    df = miw.load_labeled_csv(args.labeled)
    paired = miw.pair_sides_for_win(df, miw.DEFAULT_MAX_TDIFF)

    y_all = paired["won_1p"].astype(int).values
    groups_all = paired["video_id_1p"].values
    tsumo_vals = paired["tsumo_1p"].astype(float).values
    mid_mask = (tsumo_vals > args.fixed_q33) & (tsumo_vals <= args.fixed_q67)
    print(f"  中盤(手数 {args.fixed_q33:.0f}-{args.fixed_q67:.0f}): n={int(mid_mask.sum())} / {len(paired)}")

    indicator_cols = miw._get_indicator_cols(paired)
    feat_df = miw.build_features(paired, indicator_cols)
    X_all = feat_df.fillna(0.0).values.astype(float)
    feature_names = list(feat_df.columns)

    X_mid = X_all[mid_mask]
    y_mid = y_all[mid_mask]
    groups_mid = groups_all[mid_mask]
    n_unique = len(np.unique(groups_mid))
    folds = min(miw.N_FOLDS, max(2, n_unique))

    print(f"\n=== 中盤限定 Permutation Importance (HistGBC, {folds} fold) ===")
    perm_df = miw.compute_perm_importance_win(
        X_mid, y_mid, groups_mid, feature_names, folds
    )

    print("\n─── 中盤限定 Permutation Importance ランキング (上位20件) ───")
    for _, row in perm_df.head(20).iterrows():
        print(f"  {int(row['rank']):>3}  {row['feature']:<40}"
              f"  {row['importance_mean']:>+11.6f}  ±{row['importance_std']:>7.6f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    perm_df.to_csv(out_path, index=False)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
