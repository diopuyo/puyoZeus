"""tier1 指標の相関分析 — 冗長ペア検出 + win 信号相関 (設計用)。

目的:
  1. 正規化指標どうしの Pearson 相関 → 冗長ペア (|r|>THRESH) を洗い出し
     tier1 指標セットの確定 (削るべき重複) を判断する材料にする。
  2. 各指標と勝敗 (won) の点双列相関を overall + 位相別 (手数三分位) で出す。
     どの指標が信号を持つかをファミリー別に俯瞰する。

入力: data/indicators_v2/study/labeled_win.csv (per-side 行, won 列あり)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CSV_PATH = "data/indicators_v2/study/labeled_win.csv"
REDUNDANT_THRESH = 0.85  # |r| がこれ以上なら冗長候補

# 正規化指標 (0-1)。_raw/_source/_max_chain の補助列は除外。
NORM_INDICATORS: tuple[str, ...] = (
    # ①進行度
    "tsumo_count_rate", "board_puyo_total", "board_color_puyo_total", "margin_time_rate",
    # ②占有・危険
    "max_column_height", "column_bumpiness", "death_margin", "death_margin_neighbor",
    # ③火力・潜在
    "current_max_chain", "immediate_fire_power", "reach_fire_power", "chain_efficiency",
    "min_puyos_to_ignite", "conn_pair_count", "conn_triple_count", "conn_max_group_size",
    "second_chain_potential",
    # ④お邪魔
    "ojama_net_balance", "ojama_forecast", "board_ojama_count",
    # ⑤テンポ
    "chain_duration_sec",
    # ⑥受け力
    "dig_resistance", "absorption_capacity",
)


def _find_redundant(df: pd.DataFrame, cols: list[str]) -> list[tuple[str, str, float]]:
    """|r|>THRESH の指標ペアを (a, b, r) で返す (絶対値降順)。"""
    corr = df[cols].corr(method="pearson")
    pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= REDUNDANT_THRESH:
                pairs.append((a, b, float(r)))
    pairs.sort(key=lambda x: -abs(x[2]))
    return pairs


def _win_corr(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """各指標と won の Pearson (点双列) 相関。"""
    out = {}
    for c in cols:
        sub = df[[c, "won"]].dropna()
        if len(sub) > 10 and sub[c].std() > 0:
            out[c] = float(np.corrcoef(sub[c], sub["won"])[0, 1])
        else:
            out[c] = float("nan")
    return pd.Series(out)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    cols = [c for c in NORM_INDICATORS if c in df.columns]
    labeled = df.dropna(subset=["won"])
    print(f"全行={len(df)}  ラベル付き={len(labeled)}  指標数={len(cols)}")

    print("\n=== 1. 冗長ペア (|Pearson r| >= "
          f"{REDUNDANT_THRESH}) ===")
    pairs = _find_redundant(df, cols)
    if not pairs:
        print("  なし")
    for a, b, r in pairs:
        print(f"  r={r:+.3f}  {a}  <->  {b}")

    print("\n=== 2. win 相関 (overall + 手数三分位) ===")
    q1, q2 = labeled["tsumo"].quantile([1 / 3, 2 / 3]).tolist()
    early = labeled[labeled["tsumo"] <= q1]
    mid = labeled[(labeled["tsumo"] > q1) & (labeled["tsumo"] <= q2)]
    late = labeled[labeled["tsumo"] > q2]
    print(f"  位相境界: 序盤<={q1:.0f} 中盤<={q2:.0f} 終盤>{q2:.0f}  "
          f"(n: {len(early)}/{len(mid)}/{len(late)})")
    r_all = _win_corr(labeled, cols)
    r_e, r_m, r_l = (_win_corr(d, cols) for d in (early, mid, late))
    tbl = pd.DataFrame({"overall": r_all, "序盤": r_e, "中盤": r_m, "終盤": r_l})
    tbl["abs_late"] = tbl["終盤"].abs()
    tbl = tbl.sort_values("abs_late", ascending=False).drop(columns="abs_late")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(tbl.round(3).to_string())


if __name__ == "__main__":
    main()
