"""効率系指標と材料 (色ぷよ量) の、どちらが勝敗を予測するかを全域で測る.

## 背景 (2026-08-09 user 判定)
t=29 の盤面について user は
  「ほぼ確実に言えるのは **この盤面では色ぷよの量の差で 1P 有利**」
  「モデルがかなり未熟のため 2P 有利に見えている。 **実際は最終的な効率は
    両方変わらないレベル**」
と判定した。 しかしモデルは `飽和連鎖量 1P 0.11 / 2P 0.26`
`期待火力K1差 -0.39` と **効率で 2P 優位の差**を作っていた。

過去には [[project-saturation-ceiling-untrustworthy-2026-07-22]] で
「飽和天井は信頼不可・理想ツモ天井は空き空間量と同じで無相関」と判定済み。
それでも効率系指標が主因欄を占めている。

## 測ること
73,416 ペア (66動画) で、 各指標**単独の AUC** を出して比べる:
  - 材料系: 色ぷよ総数
  - 効率系: 飽和連鎖量 / 現在最大連鎖 / 期待火力 / 近未来火力
さらに **おじゃまフラット度で層別**して、 user 伝授の
「色ぷよはおじゃまがフラットなときに特に効く」が効率系にも当てはまるか見る。

読み取り専用。 モデルの変更はしない。
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
    _ojama_flat_score,
    _resolve_features,
)
from scripts.model_indicator_win import (  # noqa: E402
    build_features,
    load_labeled_csv,
    pair_sides_for_win,
)

# 比較する指標 (材料系 / 効率系)
MATERIAL_COLS = ("board_color_puyo_total",)
EFFICIENCY_COLS = (
    "saturated_chain_count", "current_max_chain",
    "near_future_fire_k1", "near_future_fire_k3",
    "expected_fire_k1", "sub_chain_count", "ukeyasusa",
)
FLAT_HI: float = 0.8
FLAT_LO: float = 0.37


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    """単一変数の AUC (順位法、同順位は平均化)。"""
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    if len(y) < 50 or y.sum() in (0, len(y)):
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
    flat = np.asarray(_ojama_flat_score(
        feat.get("board_ojama_count_diff", 0.0).fillna(0.0),
        feat.get("ojama_forecast_diff", 0.0).fillna(0.0),
    ), dtype=float)

    print(f"サンプル {len(y)} ペア")
    print()
    print(f"{'指標':28s} {'全体':>8s} {'ojamaフラット':>12s} {'ojama差大':>10s}")
    print("-" * 62)
    for group, cols in (("材料", MATERIAL_COLS), ("効率", EFFICIENCY_COLS)):
        for c in cols:
            col = f"{c}_diff"
            if col not in feat.columns:
                continue
            v = feat[col].fillna(0.0).values
            a_all = _auc(v, y)
            a_flat = _auc(v[flat >= FLAT_HI], y[flat >= FLAT_HI])
            a_ojama = _auc(v[flat <= FLAT_LO], y[flat <= FLAT_LO])
            print(f"[{group}] {c:22s} {a_all:8.4f} {a_flat:12.4f} {a_ojama:10.4f}")
    print()
    print("AUC 0.5 = 無相関。 0.5 より上 = その指標が大きい側が勝つ。")
    print("下 = 逆相関 (大きい側が負ける)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
