"""有利不利(win)に効く「差分・組み合わせ」を探す。

自分側の生値は弱いが、有利不利は相手との相対で決まる。
  - 生の自分値 (x_1p) の win 相関
  - 相手との差分 (x_1p - x_2p) の win 相関
  - いくつかの合成 (標準化和・相互作用) の win 相関
を比較し、「差分・組み合わせで大きく効くようになる値」を炙り出す。

入力: data/indicators_v2/study/labeled_win.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.model_indicator_win import (  # noqa: E402
    load_labeled_csv, pair_sides_for_win, _get_indicator_cols, build_features,
)

CSV_PATH = "data/indicators_v2/study/labeled_win.csv"


def _corr(x: pd.Series, y: pd.Series) -> float:
    """欠損を落とした Pearson (点双列) 相関。"""
    s = pd.concat([x, y], axis=1).dropna()
    if len(s) < 20 or s.iloc[:, 0].std() == 0:
        return float("nan")
    return float(np.corrcoef(s.iloc[:, 0], s.iloc[:, 1])[0, 1])


def _z(s: pd.Series) -> pd.Series:
    """標準化。"""
    return (s - s.mean()) / (s.std() + 1e-9)


def main() -> None:
    df = load_labeled_csv(CSV_PATH)
    paired = pair_sides_for_win(df, max_tdiff=1.0)
    cols = _get_indicator_cols(paired)
    feat = build_features(paired, cols)
    won = paired["won_1p"].astype(float)
    tsumo = paired["tsumo_1p"].astype(float)
    q2 = tsumo.quantile(2 / 3)
    late = tsumo > q2

    print(f"ペア数={len(paired)}  終盤ペア={int(late.sum())}  指標数={len(cols)}")
    print("\n=== 生の自分値 vs 相手との差分 (win_1p 相関) ===")
    print(f"{'指標':<26}{'生(全体)':>9}{'差分(全体)':>11}{'差分(終盤)':>11}")
    recs = []
    for c in cols:
        r_raw = _corr(feat[f"{c}_1p"], won)
        r_diff = _corr(feat[f"{c}_diff"], won)
        r_diff_late = _corr(feat[f"{c}_diff"][late], won[late])
        recs.append((c, r_raw, r_diff, r_diff_late))
    recs.sort(key=lambda r: -abs(r[3]) if not np.isnan(r[3]) else 0)
    for c, rr, rd, rdl in recs:
        print(f"{c:<26}{rr:>+9.3f}{rd:>+11.3f}{rdl:>+11.3f}")

    print("\n=== 合成 (標準化和・相互作用) の win_1p 相関 ===")
    d = {c: feat[f"{c}_diff"] for c in cols}
    comp: dict[str, pd.Series] = {}
    # お邪魔優位 (相手より埋まっていない) + 連鎖規模優位
    comp["お邪魔優位(=-お邪魔数diff)"] = -d["board_ojama_count"]
    comp["連鎖規模優位(現在最大連鎖diff)"] = d["current_max_chain"]
    comp["色土台優位(色ぷよ総数diff)"] = d["board_color_puyo_total"]
    # 標準化和 (お邪魔優位 + 連鎖優位 + 色土台優位)
    comp["合成A: お邪魔+連鎖+色土台"] = (
        _z(-d["board_ojama_count"]) + _z(d["current_max_chain"])
        + _z(d["board_color_puyo_total"]))
    # 相互作用: 連鎖優位 × お邪魔優位 (両方揃うと決定的)
    comp["合成B: 連鎖優位 × お邪魔優位"] = (
        _z(d["current_max_chain"]) * _z(-d["board_ojama_count"]))
    # 攻め vs 相手の受け: 自予告 - 相手掘り耐性
    comp["合成C: お邪魔予告diff + 危険diff"] = (
        _z(-d["ojama_forecast"]) + _z(d["death_margin"]))
    for name, s in comp.items():
        print(f"{name:<30} 全体={_corr(s, won):+.3f} "
              f"終盤={_corr(s[late], won[late]):+.3f}")


if __name__ == "__main__":
    main()
