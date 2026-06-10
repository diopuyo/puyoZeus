"""
ML (Logistic Regression) ベース重み学習 script

目的:
    `tune_weights.py` の grid search は 7^4 ≈ 2400 通りの離散探索のため
    最適重みを取り逃す可能性 + 4 指標しか動かせない制約がある。
    本 script は sklearn LogisticRegression で全 12 指標の連続値重みを
    L2 正則化付きで学習し、grid search との比較を行う。

特徴量:
    feature_i = p1_indicator_i - p2_indicator_i (符号付き差分)
ラベル:
    +1 if winner == "1P" else -1

実行例:
    ./venv/bin/python scripts/learn_weights_lr.py \
        --features-cache data/verify/tune_weights_v02_midpoint.json \
        --regularization 0.5 \
        --kfold 5 \
        --train-ratio 0.5 \
        --seed 42 \
        --out data/verify/learned_weights.json

    # 動画から直接抽出する場合 (重い)
    ./venv/bin/python scripts/learn_weights_lr.py \
        --video data/frames/video_02.mp4 \
        --boundaries data/verify/match_boundaries_v4/video_02/matches.tsv \
        --winners data/verify/match_winners_v02.tsv \
        --time-mode midpoint \
        --regularization 0.5 \
        --out data/verify/learned_weights.json

出力:
    data/verify/learned_weights.json
        - learned_weights : 各指標→係数 (StandardScaler は元スケールに逆変換)
        - intercept       : LR 切片
        - holdout_train_acc / holdout_test_acc
        - kfold_test_accs / kfold_test_mean / kfold_test_std
        - regularization (alpha 相当)
        - feature_names   : 学習に使った指標名 (順序固定)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# プロジェクトルートを sys.path に追加
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.old.eval_weights_holdout import (  # noqa: E402
    DEFAULT_KFOLD_K,
    DEFAULT_SEED,
    DEFAULT_TRAIN_RATIO,
    OVERFIT_GAP_THRESHOLD,
    get_samples,
    kfold_split,
    random_split,
)
from scripts.old.tune_weights import MatchSample, evaluate_weights  # noqa: E402
from src.old.scorer import DEFAULT_WEIGHTS  # noqa: E402

# ============================
# 定数
# ============================

# LR 学習対象指標 (キャッシュ JSON / IndicatorSet.results に存在するもの)
# next_acceptance は IndicatorSet 属性のみで cache に乗らないため除外。
LR_FEATURE_NAMES: tuple[str, ...] = (
    "main_chain_maturity",
    "extension_potential",
    "sub_chain_quality",
    "harassment_resistance",
    "death_risk",
    "offset_power",
    "second_chain_potential",
    "field_efficiency",
    "shape_score",
    "touching_density",
    "tail_height",
    "color_variance",
)

# LR の最大反復回数 (収束保証)
LR_MAX_ITER: int = 1000

# L2 正則化の既定強度 (sklearn の C = 1/regularization に対応)
DEFAULT_REGULARIZATION: float = 0.5


# ============================
# 特徴量行列構築
# ============================


@dataclass(frozen=True)
class FeatureMatrix:
    """指標差分行列とラベル列。"""
    X: np.ndarray  # shape=(n, d)
    y: np.ndarray  # shape=(n,), 値は +1 / -1
    feature_names: tuple[str, ...]


def build_features(
    samples: list[MatchSample],
    feature_names: tuple[str, ...] = LR_FEATURE_NAMES,
) -> FeatureMatrix:
    """各試合の (p1 - p2) 指標差分を行列化する。

    Args:
        samples: 試合サンプル。
        feature_names: 使用する指標名 (順序固定)。

    Returns:
        FeatureMatrix: X(n,d) と y(n,) +1/-1。
    """
    n = len(samples)
    d = len(feature_names)
    x_arr = np.zeros((n, d), dtype=np.float64)
    y_arr = np.zeros((n,), dtype=np.int64)
    for i, s in enumerate(samples):
        for j, name in enumerate(feature_names):
            v1 = s.p1_scores.get(name, 0.0)
            v2 = s.p2_scores.get(name, 0.0)
            x_arr[i, j] = float(v1) - float(v2)
        y_arr[i] = 1 if s.winner == "1P" else -1
    return FeatureMatrix(X=x_arr, y=y_arr, feature_names=tuple(feature_names))


# ============================
# LR 学習・評価
# ============================


@dataclass(frozen=True)
class LearnedModel:
    """学習済み LR の係数と切片 (元スケール)。

    StandardScaler の前処理を線形に巻き戻し、原指標差分に対する
    重みとして使えるようにする。
    """
    coef: np.ndarray   # shape=(d,)
    intercept: float
    feature_names: tuple[str, ...]
    train_acc: float

    def to_weights_dict(self) -> dict[str, float]:
        """指標名→重みの辞書に変換する。"""
        return {
            name: float(self.coef[i])
            for i, name in enumerate(self.feature_names)
        }


def fit_lr(
    fm: FeatureMatrix,
    regularization: float = DEFAULT_REGULARIZATION,
) -> LearnedModel:
    """L2 正則化付き LR を学習し、StandardScaler を巻き戻す。

    Args:
        fm: 特徴量行列。
        regularization: L2 正則化強度 (sklearn C = 1/regularization)。

    Returns:
        LearnedModel: 元スケール係数と切片。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(fm.X)
    c_value = 1.0 / max(regularization, 1e-9)
    clf = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=LR_MAX_ITER,
        solver="lbfgs",
    )
    clf.fit(x_scaled, fm.y)
    # StandardScaler 逆変換: w_orig_i = w_scaled_i / scale_i
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    coef_orig = clf.coef_[0] / scale
    # 切片も補正: b_orig = b_scaled - sum(w_scaled_i * mean_i / scale_i)
    intercept_orig = float(
        clf.intercept_[0]
        - np.sum(clf.coef_[0] * scaler.mean_ / scale)
    )
    train_acc = float(clf.score(x_scaled, fm.y))
    return LearnedModel(
        coef=coef_orig,
        intercept=intercept_orig,
        feature_names=fm.feature_names,
        train_acc=train_acc,
    )


def predict_with_weights(
    samples: list[MatchSample],
    weights: dict[str, float],
    intercept: float = 0.0,
) -> float:
    """重み + 切片で予測し、勝者一致率を返す。

    Args:
        samples: 評価サンプル。
        weights: 指標名→重み。
        intercept: 線形切片 (LR の bias)。

    Returns:
        accuracy (0.0〜1.0)。
    """
    if not samples:
        return 0.0
    correct = 0
    total = 0
    for s in samples:
        diff = intercept
        for name, w in weights.items():
            v1 = s.p1_scores.get(name, 0.0)
            v2 = s.p2_scores.get(name, 0.0)
            diff += (v1 - v2) * w
        if diff == 0.0:
            continue
        predicted = "1P" if diff > 0 else "2P"
        if predicted == s.winner:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


# ============================
# K-fold 評価
# ============================


def kfold_lr_scores(
    samples: list[MatchSample],
    regularization: float = DEFAULT_REGULARIZATION,
    k: int = DEFAULT_KFOLD_K,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """K-fold で LR を学習・評価する。

    Returns:
        各 fold の train/test 精度と平均/標準偏差を含む辞書。
    """
    folds = kfold_split(samples, k=k, seed=seed)
    fold_results: list[dict[str, Any]] = []
    for i, (train, test) in enumerate(folds):
        fm_train = build_features(train)
        fm_test = build_features(test)
        model = fit_lr(fm_train, regularization=regularization)
        weights = model.to_weights_dict()
        test_acc = predict_with_weights(
            test, weights, intercept=model.intercept,
        )
        fold_results.append({
            "fold": i,
            "n_train": len(train),
            "n_test": len(test),
            "train_acc": model.train_acc,
            "test_acc": test_acc,
            "weights": weights,
            "intercept": model.intercept,
        })
    test_accs = [f["test_acc"] for f in fold_results]
    return {
        "k": k,
        "seed": seed,
        "fold_results": fold_results,
        "test_mean": statistics.fmean(test_accs),
        "test_std": (
            statistics.pstdev(test_accs) if len(test_accs) > 1 else 0.0
        ),
        "train_mean": statistics.fmean(
            [f["train_acc"] for f in fold_results]
        ),
    }


# ============================
# 統合評価 + レポート
# ============================


def run_holdout_lr(
    samples: list[MatchSample],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    regularization: float = DEFAULT_REGULARIZATION,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """train/test split で LR を学習・評価する。"""
    split = random_split(
        samples, train_ratio=train_ratio, seed=seed,
    )
    fm_train = build_features(split.train)
    model = fit_lr(fm_train, regularization=regularization)
    weights = model.to_weights_dict()
    test_acc = predict_with_weights(
        split.test, weights, intercept=model.intercept,
    )
    gap = model.train_acc - test_acc
    return {
        "n_train": len(split.train),
        "n_test": len(split.test),
        "train_acc": model.train_acc,
        "test_acc": test_acc,
        "generalization_gap": gap,
        "overfit_flag": gap >= OVERFIT_GAP_THRESHOLD,
        "weights": weights,
        "intercept": model.intercept,
        "regularization": regularization,
    }


def print_lr_report(
    holdout: dict[str, Any], kfold: dict[str, Any],
) -> None:
    """LR レポートを stdout に整形出力する。"""
    print("\n========== LR 重み学習レポート ==========")
    print(f"holdout: train={holdout['train_acc']:.3f}, "
          f"test={holdout['test_acc']:.3f}, "
          f"gap={holdout['generalization_gap']:+.3f}, "
          f"overfit={holdout['overfit_flag']}")
    print(f"K-fold (k={kfold['k']}): "
          f"train_mean={kfold['train_mean']:.3f}, "
          f"test_mean={kfold['test_mean']:.3f}, "
          f"test_std={kfold['test_std']:.3f}")
    print("\n[learned weights, holdout]")
    for name, val in sorted(
        holdout["weights"].items(), key=lambda x: -abs(x[1]),
    ):
        print(f"  {name:<25s} {val:+.4f}")
    print(f"  intercept                 {holdout['intercept']:+.4f}")


# ============================
# main エントリ
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LR (sklearn) ベース重み学習 + K-fold CV",
    )
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument(
        "--winners", type=Path,
        default=Path("data/verify/match_winners_v02.tsv"),
    )
    parser.add_argument(
        "--boundaries", type=Path,
        default=Path("data/verify/match_boundaries_v4/video_02/matches.tsv"),
    )
    parser.add_argument(
        "--features-cache", type=Path, default=None,
        help="tune_weights.py の出力 JSON。指定時は動画を読まない",
    )
    parser.add_argument(
        "--time-mode", choices=("end", "midpoint"), default="midpoint",
    )
    parser.add_argument("--offset-sec", type=float, default=3.0)
    parser.add_argument(
        "--regularization", type=float, default=DEFAULT_REGULARIZATION,
    )
    parser.add_argument(
        "--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO,
    )
    parser.add_argument("--kfold", type=int, default=DEFAULT_KFOLD_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/verify/learned_weights.json"),
    )
    args = parser.parse_args()

    samples = get_samples(args)

    holdout = run_holdout_lr(
        samples,
        train_ratio=args.train_ratio,
        regularization=args.regularization,
        seed=args.seed,
    )
    kfold = kfold_lr_scores(
        samples,
        regularization=args.regularization,
        k=args.kfold,
        seed=args.seed,
    )

    out_payload = {
        "n_samples": len(samples),
        "feature_names": list(LR_FEATURE_NAMES),
        "regularization": args.regularization,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "holdout": holdout,
        "kfold": kfold,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_lr_report(holdout, kfold)
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
