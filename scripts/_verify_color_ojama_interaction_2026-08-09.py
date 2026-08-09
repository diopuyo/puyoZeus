"""交互作用 (色ぷよ差 × おじゃまフラット度) の効果を学習データで検証する.

user 伝授 (2026-08-09):
  「色ぷよはお邪魔状況や予告お邪魔などがフラットな時に特に有利不利に
    優位にうごきます」

検証したいこと:
  1. **知見そのものが実データで成り立つか** — おじゃまがフラットな局面に限れば、
     色ぷよ差と勝敗の関係が強くなるか (層別して相関/AUC を見る)
  2. 交互作用列を足すとモデルの予測性能が上がるか (AUC の比較)

過学習防止則に従い、 デモの 1 シーンではなく **学習データ全体を層別**して見る。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.visualize_advantage_overlay import (  # noqa: E402
    OJAMA_FLAT_SCALE,
    _ojama_flat_score,
    _resolve_features,
)
from scripts.model_indicator_win import (  # noqa: E402
    build_features,
    load_labeled_csv,
    pair_sides_for_win,
)

CSV = "data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv"
# フラット度でこの値以上を「フラットな局面」とみなす (層別の閾値)。
# exp(-1)≒0.37 は「おじゃま差の合計が OJAMA_FLAT_SCALE 個」に対応する。
FLAT_HI: float = 0.8   # ほぼ差なし
FLAT_LO: float = 0.37  # おじゃま差が大きい


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    """単一変数の AUC を順位法で厳密に計算する (sklearn 非依存)."""
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # 同順位の平均化
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    df = load_labeled_csv(CSV)
    feat_cols = _resolve_features(df)
    paired = pair_sides_for_win(df, max_tdiff=1.0)
    feat = build_features(paired, feat_cols)
    y = paired["won_1p"].astype(int).values

    need = ("board_color_puyo_total_diff", "board_ojama_count_diff",
            "ojama_forecast_diff")
    for c in need:
        if c not in feat.columns:
            print(f"必要列が無い: {c}")
            return 1

    color = feat["board_color_puyo_total_diff"].fillna(0.0).values
    flat = np.asarray(_ojama_flat_score(
        feat["board_ojama_count_diff"].fillna(0.0),
        feat["ojama_forecast_diff"].fillna(0.0),
    ), dtype=float)

    print(f"サンプル数 {len(y)} / フラット度スケール {OJAMA_FLAT_SCALE} (正規化値)")
    print(f"フラット度 中央値 {np.median(flat):.3f} / "
          f">= {FLAT_HI} が {(flat >= FLAT_HI).mean():.1%} / "
          f"<= {FLAT_LO} が {(flat <= FLAT_LO).mean():.1%}")
    print()
    print("=== 色ぷよ差 単独の AUC (勝敗予測) を おじゃまフラット度で層別 ===")
    for name, mask in (
        (f"フラット (>= {FLAT_HI})", flat >= FLAT_HI),
        (f"中間", (flat > FLAT_LO) & (flat < FLAT_HI)),
        (f"差が大きい (<= {FLAT_LO})", flat <= FLAT_LO),
        ("全体", np.ones(len(y), dtype=bool)),
    ):
        n = int(mask.sum())
        if n < 100:
            print(f"  {name:22s} n={n:6d}  (サンプル不足)")
            continue
        a = _auc(color[mask], y[mask])
        print(f"  {name:22s} n={n:6d}  AUC={a:.4f}")
    print()
    print("=== 交互作用列 単独の AUC (全体) ===")
    inter = color * flat
    print(f"  色ぷよ差 単独           AUC={_auc(color, y):.4f}")
    print(f"  色ぷよ差 × フラット度   AUC={_auc(inter, y):.4f}")
    print()
    print("解釈: フラット層の AUC が 差が大きい層より高ければ user 伝授の裏付け。")
    print("      交互作用列の AUC が色ぷよ差単独より高ければ特徴として有効。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
