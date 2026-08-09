"""おじゃまフラット度を特徴に加えるとモデルの予測が良くなるかを検証する.

## 何を比べるか
  A: 従来の特徴量のみ
  B: A + 色ぷよ差×フラット度 (掛け算の交互作用)
  C: B + フラット度そのもの  ← 木が局面別の分岐を学習できる

## なぜ C が要ると考えたか (実測、73,416ペア)
おじゃまがフラットな局面では効率系指標が全て AUC 0.49〜0.50 (無相関) になり、
色ぷよ総数だけが 0.5380 で効く。 逆におじゃま差が大きい局面では効率系が
0.65〜0.75 で効く。 **局面によって効く軸が入れ替わる**ため、 モデルに
「今どちらの局面か」を教える列が要る。

## 評価方法
**動画ホールドアウト** (動画単位で分割) で AUC を出す。 同一動画の行が
学習と検証に混ざるとリークするため
([[project-cnn-regen-fair-comparison-2026-08-01]] の教訓)。
AUC は sklearn を使わず順位法で厳密計算する (過去に近似バグで誤結論を出した
ため、 [[project-cnn-regen-fair-comparison-2026-08-01]])。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from scripts.visualize_advantage_overlay import (  # noqa: E402
    COLOR_OJAMA_INTERACTION_COL,
    GBC_PARAMS,
    OJAMA_FLAT_COL,
    TRAIN_CSV_PATH,
    _ojama_flat_score,
    _resolve_features,
)
from scripts.model_indicator_win import (  # noqa: E402
    build_features,
    load_labeled_csv,
    pair_sides_for_win,
)

N_FOLDS: int = 4


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fit_eval(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """動画ホールドアウトで AUC を出す (対称化は学習側のみ)。"""
    vids = np.unique(groups)
    rng = np.random.RandomState(0)
    rng.shuffle(vids)
    folds = np.array_split(vids, N_FOLDS)
    aucs = []
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        if te.sum() < 100 or tr.sum() < 100:
            continue
        Xtr, ytr = X[tr], y[tr]
        # 対称化 (差分は側を入れ替えると符号反転する反対称関数)
        Xs = np.vstack([Xtr, -Xtr])
        ys = np.concatenate([ytr, 1 - ytr])
        m = HistGradientBoostingClassifier(**GBC_PARAMS)
        m.fit(Xs, ys)
        p = m.predict_proba(X[te])[:, 1]
        a = _auc(p, y[te])
        if not np.isnan(a):
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


def main() -> int:
    df = load_labeled_csv(TRAIN_CSV_PATH)
    feat_cols = _resolve_features(df)
    paired = pair_sides_for_win(df, max_tdiff=1.0)
    feat = build_features(paired, feat_cols)
    y = paired["won_1p"].astype(int).values
    groups = paired["video_id_1p"].astype(str).values

    base_cols = [f"{c}_diff" for c in feat_cols]
    flat = np.asarray(_ojama_flat_score(
        feat["board_ojama_count_diff"].fillna(0.0),
        feat["ojama_forecast_diff"].fillna(0.0),
    ), dtype=float)
    inter = feat["board_color_puyo_total_diff"].fillna(0.0).values * flat

    XA = feat[base_cols].fillna(0.0).values
    XB = np.column_stack([XA, inter])
    XC = np.column_stack([XB, flat])

    print(f"サンプル {len(y)} ペア / 動画 {len(np.unique(groups))} 本 "
          f"/ {N_FOLDS}分割 動画ホールドアウト")
    print()
    for name, X in (("A 従来のみ", XA),
                    ("B A+色ぷよ×フラット度", XB),
                    ("C B+フラット度そのもの", XC)):
        a = _fit_eval(X, y, groups)
        print(f"  {name:26s} AUC = {a:.4f}")
    print()
    print("差が 0.005 未満なら誤差の範囲とみなす (単一 run の揺らぎ)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
