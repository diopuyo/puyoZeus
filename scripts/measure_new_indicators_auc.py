"""条件2(潰し)+ C(形品質)新指標の位相別AUCを測定する。

labeled_win.csv(再収集+label後)から、各行の指標値(片側の絶対値)と won の
単変量AUCを 序盤/中盤/終盤(tsumo_count_rate で3分割)ごとに算出。
中盤最強の conn_pair_count(既測 0.562)と比較して効くかを見る。
逆相関の指標は 1-AUC(flip)も併記。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CSV = "data/indicators_v2/study/labeled_win.csv"
# 測定対象(新指標 + 比較用の既存中盤最強/終盤最強)
TARGETS = [
    "ojama_disruption", "main_linked_pair_count", "isolated_pair_count",
    "main_linked_ratio", "conn_pair_count", "board_ojama_count", "current_max_chain",
]
PHASES = [("序盤", 0.0, 0.34), ("中盤", 0.34, 0.67), ("終盤", 0.67, 1.01)]


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    """単変量AUC。分散ゼロ/片クラスは nan。"""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30 or len(set(y[m].tolist())) < 2 or np.std(x[m]) == 0:
        return float("nan")
    return roc_auc_score(y[m], x[m])


def main() -> None:
    df = pd.read_csv(CSV)
    df = df[df["won"].notna()].copy()
    df["won"] = df["won"].astype(int)
    rate = df.get("tsumo_count_rate")
    if rate is None:
        print("tsumo_count_rate 列なし。中止"); return
    print(f"行数 {len(df)} / won=1 {int(df['won'].sum())}")
    header = f"{'指標':<24}" + "".join(f"{p[0]:>10}" for p in PHASES) + f"{'全体':>10}"
    print(header)
    for col in TARGETS:
        if col not in df.columns or df[col].notna().sum() < 30:
            print(f"{col:<24}{'(列/データ無)':>40}"); continue
        cells = []
        for _, lo, hi in PHASES:
            seg = df[(rate >= lo) & (rate < hi)]
            a = _auc(seg["won"].to_numpy(float), seg[col].to_numpy(float))
            cells.append(a)
        all_auc = _auc(df["won"].to_numpy(float), df[col].to_numpy(float))
        row = f"{col:<24}" + "".join(f"{c:>10.3f}" for c in cells) + f"{all_auc:>10.3f}"
        print(row)
    print("\n※AUC<0.5は逆相関(flip=1-AUC)。中盤 conn_pair 0.562 を超えるかが焦点。")


if __name__ == "__main__":
    main()
