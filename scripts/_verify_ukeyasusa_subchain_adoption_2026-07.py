"""受けやすさ(ukeyasusa)+副砲(sub_chain_count)の overlay モデル採用効果を再検証する。

## 背景
Round10 の再帰属で中盤 marginal 寄与が確定した2指標
(ukeyasusa: +0.033, sub_chain_count: +0.014) を
scripts/visualize_advantage_overlay.py の FEATURE_CANDIDATES に正式採用した。
本スクリプトは overlay の学習経路(diff特徴のみ・対称化なしの素の OOF AUC)を
模して、以下4構成の中盤/序盤/終盤 OOF win-AUC を比較する:

  1) baseline12      : FEATURES(12指標)のみ (旧来の overlay モデル相当)
  2) +saturated       : baseline + saturated_chain_count
                         (labeled_win.csv に列が既にあるため実は現状の
                          _train_model() で既に暗黙採用されている構成。
                          比較の透明性のため明示的に切り出す)
  3) +ukey_sub        : baseline + ukeyasusa + sub_chain_count (本タスクの採用対象)
  4) +all4            : baseline + saturated + ukeyasusa + sub_chain_count

## 使い方
    PYTHONPATH=. ./venv/bin/python -m scripts._verify_ukeyasusa_subchain_adoption_2026-07
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.model_indicator_win import (  # noqa: E402
    N_FOLDS, TSUMO_EARLY_RATIO, TSUMO_LATE_RATIO,
    load_labeled_csv, pair_sides_for_win, build_features, run_oof_classifier,
)
from scripts.visualize_advantage_overlay import FEATURES  # noqa: E402

LABELED_PATH = "data/indicators_v2/study/labeled_win.csv"
MAX_TDIFF = 1.0

# 検証する4構成 (指標名の候補追加のみ、FEATURES(12) は不変)
CONFIGS: dict[str, tuple[str, ...]] = {
    "baseline12": tuple(FEATURES),
    "+saturated": tuple(FEATURES) + ("saturated_chain_count",),
    "+ukey_sub": tuple(FEATURES) + ("ukeyasusa", "sub_chain_count"),
    "+all4": tuple(FEATURES) + ("saturated_chain_count", "ukeyasusa", "sub_chain_count"),
}


def _phase_masks(paired: pd.DataFrame) -> dict[str, np.ndarray]:
    """全構成で共通のtsumo三分位境界を使い、序盤/中盤/終盤マスクを返す。"""
    tsumo = paired["tsumo_1p"].astype(float).values
    q33 = float(np.quantile(tsumo, TSUMO_EARLY_RATIO))
    q67 = float(np.quantile(tsumo, TSUMO_LATE_RATIO))
    return {
        "序盤": tsumo <= q33,
        "中盤": (tsumo > q33) & (tsumo <= q67),
        "終盤": tsumo > q67,
    }


def _auc_diff_only(
    paired: pd.DataFrame, feat_cols: tuple[str, ...], y: np.ndarray,
    groups: np.ndarray, mask: np.ndarray | None = None,
) -> tuple[float, int]:
    """diff特徴のみ(overlay と同一構成)で GroupKFold OOF AUC を計算する。"""
    feat = build_features(paired, list(feat_cols))
    cols = [f"{c}_diff" for c in feat_cols]
    X = feat[cols].fillna(0.0).values
    yy, gg = y, groups
    if mask is not None:
        X, yy, gg = X[mask], y[mask], groups[mask]
    n_unique = len(np.unique(gg))
    folds = min(N_FOLDS, max(2, n_unique))
    if len(X) < 20 or len(np.unique(yy)) < 2:
        return float("nan"), len(X)
    oof, _ = run_oof_classifier(X, yy, gg, folds)
    valid = ~np.isnan(oof[:, 0])
    auc = float(roc_auc_score(yy[valid], oof[valid, 1]))
    return auc, int(valid.sum())


def main() -> None:
    print("=== データ読み込み ===")
    df = load_labeled_csv(LABELED_PATH)
    paired = pair_sides_for_win(df, MAX_TDIFF)
    y = paired["won_1p"].astype(int).values
    groups = paired["video_id_1p"].values
    masks = _phase_masks(paired)

    print()
    print("=" * 78)
    print("  受けやすさ+副砲 採用効果検証 (overlay diff特徴のみ, GroupKFold OOF AUC)")
    print("=" * 78)
    header = f"  {'構成':<12}  {'全体':>7}  " + "  ".join(f"{p:>7}" for p in masks)
    print(header)
    print("  " + "-" * (len(header) - 2))

    results: dict[str, dict[str, float]] = {}
    for name, cols in CONFIGS.items():
        row: dict[str, float] = {}
        auc_all, n_all = _auc_diff_only(paired, cols, y, groups)
        row["全体"] = auc_all
        line = f"  {name:<12}  {auc_all:>7.4f}  "
        for phase, mask in masks.items():
            auc_p, n_p = _auc_diff_only(paired, cols, y, groups, mask)
            row[phase] = auc_p
            line += f"{auc_p:>7.4f}  "
        print(line)
        results[name] = row

    print()
    print("  ─── 差分 (baseline12 比) ───")
    base = results["baseline12"]
    for name in ("+saturated", "+ukey_sub", "+all4"):
        deltas = " ".join(
            f"{p}:{results[name][p] - base[p]:+.4f}" for p in ["全体", *masks]
        )
        print(f"  {name:<12}  {deltas}")


if __name__ == "__main__":
    main()
