"""Phase 1-3 主因除外リストの根拠を取る (読み取り専用、2026-08-11).

`scripts/_verify_efficiency_vs_material_2026-08-09.py` は expected_fire_k1 のみ
検証していた。 本スクリプトは expected_fire_k2 を含む主因欄の全候補指標について
同じ方式 (単独変数 AUC、全体/おじゃまフラット/おじゃま差大の3層) で測り直し、
`production_config.ATTRIBUTION_EXCLUDED_INDICATORS` の根拠数値を確定する。

モデルの変更はしない。 学習・推論経路は一切触らない。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.visualize_advantage_overlay import (  # noqa: E402
    JP_LABEL,
    TRAIN_CSV_PATH,
    _ojama_flat_score,
    _resolve_features,
)
from scripts.model_indicator_win import (  # noqa: E402
    build_features,
    load_labeled_csv,
    pair_sides_for_win,
)

FLAT_HI: float = 0.8
FLAT_LO: float = 0.37


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    """単一変数の AUC (順位法、同順位は平均化)。

    `_verify_efficiency_vs_material_2026-08-09.py` の同名関数と同一実装
    (ファイル名にハイフンを含み import 不可のためコピー、ロジック変更禁止)。
    """
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
# JP_LABEL に載っている「主因表示に出得る全候補」を対象にする
# (board_ojama_count 等の材料/おじゃま系は Phase1-1 で有効性確定済みのため除外対象からは外し、
#  無情報判定が疑われるものだけ確認する)。
CHECK_COLS: tuple[str, ...] = (
    "expected_fire_k1", "expected_fire_k2",
    "saturated_chain_count", "current_max_chain",
)


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
    for c in CHECK_COLS:
        col = f"{c}_diff"
        if col not in feat.columns:
            print(f"{c:28s} 列なし (未収集)")
            continue
        v = feat[col].fillna(0.0).values
        a_all = _auc(v, y)
        a_flat = _auc(v[flat >= FLAT_HI], y[flat >= FLAT_HI])
        a_ojama = _auc(v[flat <= FLAT_LO], y[flat <= FLAT_LO])
        label = JP_LABEL.get(c, c)
        print(f"{label:28s} {a_all:8.4f} {a_flat:12.4f} {a_ojama:10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
