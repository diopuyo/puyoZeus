"""連鎖シミュレーション由来の指標が「位相別」に生きているかを測る (2026-08-09).

## user 確認事項
> 連鎖シミュレーションによる中盤評価って生きてますか?

## これまでの測定の限界
2026-08-09 の測定は「おじゃま差の有無 (フラット度)」で層別したもので、
**位相 (序盤/中盤/終盤) で切っていない**。 おじゃまフラット ≒ 中盤が多い、
という対応はあるが厳密ではないため、 位相で切り直す。

位相の切り方は確定知見に従い **試合内の相対進行率** を使う
([[project-win-eval-regen-2026-07-26]]: 「位相は試合内相対進行率で切るべき」)。
絶対時刻で切ると試合長のばらつきに引きずられる。

## 測る指標 (すべて内部で連鎖シミュレーションを回す)
current_max_chain / saturated_chain_count / near_future_fire_k1..5 /
expected_fire_k1,k2 / sub_chain_count

比較対象として、 シミュレーションを使わない指標も併記する
(board_ojama_count / board_color_puyo_total / ukeyasusa / conn_*)。

読み取り専用。 AUC は順位法で厳密計算 (過去に近似バグで誤結論を出したため)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.visualize_advantage_overlay import (  # noqa: E402
    TRAIN_CSV_PATH,
    _resolve_features,
)
from scripts.model_indicator_win import (  # noqa: E402
    build_features,
    load_labeled_csv,
    pair_sides_for_win,
)

# 連鎖シミュレーションを回す指標
SIM_COLS = (
    "current_max_chain", "saturated_chain_count", "sub_chain_count",
    "near_future_fire_k1", "near_future_fire_k3", "near_future_fire_k5",
    "expected_fire_k1", "fire_stability_k4",
)
# シミュレーションを使わない指標 (比較対象)
NON_SIM_COLS = (
    "board_ojama_count", "board_color_puyo_total", "ukeyasusa",
    "conn_triple_count", "max_column_height", "death_margin",
)
# 位相の区切り (試合内相対進行率)
PHASES = (("序盤", 0.0, 0.33), ("中盤", 0.33, 0.66), ("終盤", 0.66, 1.01))


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    if len(y) < 200 or y.sum() in (0, len(y)):
        return float("nan")
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    df = load_labeled_csv(TRAIN_CSV_PATH)
    feat_cols = _resolve_features(df)
    paired = pair_sides_for_win(df, max_tdiff=1.0)
    feat = build_features(paired, feat_cols)
    y = paired["won_1p"].astype(int).values

    # 試合内の相対進行率を出す (video_id × game_idx ごとに正規化)
    t = paired["t_sec_1p"].astype(float).values
    vid = paired["video_id_1p"].astype(str).values
    gid = paired["game_idx_1p"].astype(int).values
    rel = np.zeros(len(t), dtype=float)
    keys = np.char.add(np.char.add(vid, "#"), gid.astype(str))
    for k in np.unique(keys):
        m = keys == k
        lo, hi = t[m].min(), t[m].max()
        rel[m] = 0.5 if hi <= lo else (t[m] - lo) / (hi - lo)

    print(f"サンプル {len(y)} ペア / 試合 {len(np.unique(keys))}")
    for name, lo, hi in PHASES:
        n = int(((rel >= lo) & (rel < hi)).sum())
        print(f"  {name}: {n} ペア")
    print()
    header = f"{'指標':26s}" + "".join(f"{p[0]:>10s}" for p in PHASES) + f"{'全体':>10s}"
    print(header)
    print("-" * len(header))
    for group, cols in (("[シミュ]", SIM_COLS), ("[非シミュ]", NON_SIM_COLS)):
        for c in cols:
            col = f"{c}_diff"
            if col not in feat.columns:
                continue
            v = feat[col].fillna(0.0).values
            cells = []
            for _, lo, hi in PHASES:
                m = (rel >= lo) & (rel < hi)
                cells.append(f"{_auc(v[m], y[m]):10.4f}")
            cells.append(f"{_auc(v, y):10.4f}")
            print(f"{group}{c:18s}" + "".join(cells))
    print()
    print("AUC 0.5 = 無相関。 0.5 から離れているほど勝敗を予測できている。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
