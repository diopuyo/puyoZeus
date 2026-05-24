"""
段階 3: 多重共線性除去 + クリーンな重み学習 (v3)

目的:
    learn_weights_v2 の counter-intuitive 重み (offset_power 負, etc.) は
    主要因が多重共線性 (VIF > 10 多数 + 高相関ペア) と推定された。
    v3 では:
        1. 多重共線性除去 (VIF >10 / |r|>=0.85 ペアの片方を削除)
        2. 1380 サンプル (v2 csv) で複数モデル比較
        3. 物理的に正の指標を「正値制約」して符号反転を抑止

モデル比較:
    - LR L2 (C 探索)
    - LR L1 (sparsity)
    - Elastic Net (multicollinearity 耐性)
    - Ridge Classifier (constraint で重み平滑化)
    - Random Forest (feature importance)
    - 正値制約 LR (Ridge with positive=True / sklearn 1.4+)

検証:
    - Train (動画 01+02) / Test (動画 03) split
    - K-fold (k=5)
    - Per-time-phase 精度

出力:
    - data/verify/learned_weights_v3.json
    - 完了後、src/scorer.py に LEARNED_WEIGHTS_V3_GLOBAL を追記
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# sklearn 警告抑制
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn",
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn",
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning,
)

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.eda_features import Dataset, load_dataset  # noqa: E402
from scripts.generate_training_dataset import FEATURE_NAMES  # noqa: E402
from scripts.learn_weights_v2 import (  # noqa: E402
    LR_L1_C_GRID,
    LR_L2_C_GRID,
    LR_MAX_ITER,
    RANDOM_SEED,
    RF_MAX_DEPTH_GRID,
    RF_N_ESTIMATORS,
    ModelResult,
    Split,
    accuracy,
    default_weights_vector,
    fit_lr_eval,
    fit_rf_eval,
    normalize_weights_to_default_scale,
    predict_with_weights,
    random_split,
    serialize_result,
    video_level_split,
)
from scripts.multicollinearity_analysis import (  # noqa: E402
    VIF_THRESHOLD_SEVERE,
    compute_vif,
)
from src.scorer import DEFAULT_WEIGHTS  # noqa: E402

# ============================
# 定数
# ============================

DEFAULT_INPUT_CSV: Path = Path("data/training/match_features_v2.csv")
DEFAULT_OUTPUT_JSON: Path = Path("data/verify/learned_weights_v3.json")

# 多重共線性除去で削除する特徴量 (段階 1 の結論):
#   - next_acceptance: VIF=inf (盤面差分がほぼ常に 0、無情報)
#   - offset_power: VIF=19.59、main_chain_maturity と r=0.84 で重複
#   - touching_density: VIF=6.91、offset_power と r=0.76 で重複
# main_chain_maturity は重要なので残し、その他高 VIF を削除。
DROPPED_FEATURES: tuple[str, ...] = (
    "next_acceptance",
    "offset_power",
    "touching_density",
)

# 物理的に「正の貢献」が期待される指標 (正値制約に使用)
PHYSICALLY_POSITIVE_FEATURES: tuple[str, ...] = (
    "main_chain_maturity",
    "extension_potential",
    "sub_chain_quality",
    "harassment_resistance",
    "second_chain_potential",
    "field_efficiency",
    "shape_score",
    "key_flexibility",
    "sub_chain_independence",
    "chain_timing_pressure",
)

# 物理的に「負の貢献」が期待される指標
PHYSICALLY_NEGATIVE_FEATURES: tuple[str, ...] = (
    "death_risk",
)

# Elastic Net の l1_ratio
EN_L1_RATIO_GRID: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)
EN_C_GRID: tuple[float, ...] = (0.1, 0.5, 1.0, 5.0)
KFOLD_K: int = 5


# ============================
# 特徴量フィルタ
# ============================


@dataclass(frozen=True)
class ReducedDataset:
    """共線性除去後のデータセット。"""
    feature_names: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    video_ids: list[str]
    time_phases: list[str]
    dropped: tuple[str, ...]


def reduce_features(
    ds: Dataset, dropped: tuple[str, ...] = DROPPED_FEATURES,
) -> ReducedDataset:
    """指定特徴量を削除した縮小データセットを返す。"""
    keep_idx = [
        i for i, n in enumerate(ds.feature_names) if n not in dropped
    ]
    keep_names = tuple(ds.feature_names[i] for i in keep_idx)
    return ReducedDataset(
        feature_names=keep_names,
        X=ds.X[:, keep_idx],
        y=ds.y,
        video_ids=list(ds.video_ids),
        time_phases=list(ds.time_phases),
        dropped=tuple(dropped),
    )


def reduced_to_dataset(rd: ReducedDataset) -> Dataset:
    """ReducedDataset を eda_features.Dataset 互換に変換する。"""
    return Dataset(
        feature_names=rd.feature_names,
        X=rd.X, y=rd.y,
        video_ids=rd.video_ids,
        time_phases=rd.time_phases,
    )


# ============================
# Elastic Net / Ridge Classifier
# ============================


def fit_elastic_net(
    ds: Dataset, split: Split, l1_ratio: float, C: float,
) -> ModelResult:
    """Elastic Net (LR with elasticnet penalty) を学習する。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_tr, y_tr = ds.X[split.train_idx], ds.y[split.train_idx]
    X_te, y_te = ds.X[split.test_idx], ds.y[split.test_idx]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_tr)
    clf = LogisticRegression(
        penalty="elasticnet", C=C, solver="saga",
        l1_ratio=l1_ratio, max_iter=LR_MAX_ITER,
        class_weight="balanced",
    )
    clf.fit(Xs, y_tr)
    train_acc = float(clf.score(Xs, y_tr))
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    coef_orig = clf.coef_[0] / scale
    intercept_orig = float(
        clf.intercept_[0] - np.sum(clf.coef_[0] * scaler.mean_ / scale),
    )
    pred_te = predict_with_weights(X_te, coef_orig, intercept_orig)
    test_acc = accuracy(pred_te, y_te)
    weights = {n: float(coef_orig[i]) for i, n in enumerate(ds.feature_names)}
    return ModelResult(
        model="elasticnet",
        params={"C": C, "l1_ratio": l1_ratio},
        split=split.label,
        train_acc=train_acc, test_acc=test_acc,
        weights=weights, intercept=intercept_orig,
    )


def fit_ridge_classifier(
    ds: Dataset, split: Split, alpha: float = 1.0,
) -> ModelResult:
    """RidgeClassifier (closed-form L2) を学習する。"""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler

    X_tr, y_tr = ds.X[split.train_idx], ds.y[split.train_idx]
    X_te, y_te = ds.X[split.test_idx], ds.y[split.test_idx]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_tr)
    clf = RidgeClassifier(alpha=alpha, class_weight="balanced")
    clf.fit(Xs, y_tr)
    train_acc = float(clf.score(Xs, y_tr))
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    coef_orig = clf.coef_[0] / scale
    intercept_orig = float(
        clf.intercept_[0] - np.sum(clf.coef_[0] * scaler.mean_ / scale),
    )
    pred_te = predict_with_weights(X_te, coef_orig, intercept_orig)
    test_acc = accuracy(pred_te, y_te)
    weights = {n: float(coef_orig[i]) for i, n in enumerate(ds.feature_names)}
    return ModelResult(
        model="ridge_clf",
        params={"alpha": alpha},
        split=split.label,
        train_acc=train_acc, test_acc=test_acc,
        weights=weights, intercept=intercept_orig,
    )


# ============================
# 物理符号制約付き LR (NNLS ベース)
# ============================


def fit_sign_constrained_logreg(
    ds: Dataset, split: Split, alpha: float = 1.0,
) -> ModelResult:
    """物理符号制約付き Logistic 風モデル。

    sklearn LogisticRegression は positive 制約非対応のため、
    PHYSICALLY_NEGATIVE_FEATURES の符号を反転して全体を正値制約問題に
    変換し、scipy.optimize.nnls (非負最小二乗) で解く。

    線形モデル z = X @ w + b でラベル ±1 の符号一致を最大化する近似として、
    L2 正則化 NNLS で w (>=0) を求め、b は y との残差平均で決める。
    """
    from scipy.optimize import nnls
    from sklearn.preprocessing import StandardScaler

    X_tr, y_tr = ds.X[split.train_idx], ds.y[split.train_idx]
    X_te, y_te = ds.X[split.test_idx], ds.y[split.test_idx]
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    # 物理的に負の特徴量は符号反転 (NNLS は w>=0 のみ解く)
    sign = np.array([
        -1.0 if n in PHYSICALLY_NEGATIVE_FEATURES else 1.0
        for n in ds.feature_names
    ])
    sign_pos_mask = np.array([
        n in PHYSICALLY_POSITIVE_FEATURES or n in PHYSICALLY_NEGATIVE_FEATURES
        for n in ds.feature_names
    ])
    # 制約対象列のみ符号反転で正に揃える
    Xs_signed = Xs_tr * sign[None, :]
    # L2 正則化を NNLS の式に組み込む: argmin ||A w - b||^2
    # 拡張行列で alpha I を下に積む
    n, d = Xs_signed.shape
    aug_A = np.vstack([Xs_signed, np.sqrt(alpha) * np.eye(d)])
    aug_b = np.concatenate([y_tr.astype(np.float64), np.zeros(d)])
    w_signed, _ = nnls(aug_A, aug_b, maxiter=10000)
    # 制約対象でない列 (中立) は別途 LR で許容したいが、簡易のため一律 NNLS
    w_orig_scale = w_signed * sign / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    # 符号制約対象でない指標は元 LR と同等の解として、ridge_clf の重みで穏やかに置換
    if (~sign_pos_mask).any():
        ridge = fit_ridge_classifier(ds, split, alpha=alpha)
        for i, n in enumerate(ds.feature_names):
            if not sign_pos_mask[i]:
                w_orig_scale[i] = ridge.weights[n]
    intercept = float(
        -np.mean(Xs_tr @ (w_signed * sign)) + np.mean(y_tr),
    )
    pred_tr = predict_with_weights(X_tr, w_orig_scale, intercept)
    pred_te = predict_with_weights(X_te, w_orig_scale, intercept)
    return ModelResult(
        model="sign_constrained",
        params={"alpha": alpha, "constrained_n": int(sign_pos_mask.sum())},
        split=split.label,
        train_acc=accuracy(pred_tr, y_tr),
        test_acc=accuracy(pred_te, y_te),
        weights={n: float(w_orig_scale[i]) for i, n in enumerate(ds.feature_names)},
        intercept=intercept,
    )


# ============================
# K-fold
# ============================


def kfold_eval(
    ds: Dataset, k: int, model_fn: Any, **kwargs: Any,
) -> dict[str, Any]:
    """サンプル単位 K-fold で test_acc 平均を計算する。"""
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(ds.y)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    accs: list[float] = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        split = Split(
            train_idx=train_idx, test_idx=test_idx, label=f"kfold_{i}",
        )
        res = model_fn(ds, split, **kwargs)
        accs.append(res.test_acc)
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "folds": [float(a) for a in accs]}


# ============================
# Per-phase eval
# ============================


def per_phase_acc(
    ds: Dataset, weights: np.ndarray, intercept: float = 0.0,
) -> dict[str, dict[str, float]]:
    """time_phase 別の test_acc を返す。"""
    out: dict[str, dict[str, float]] = {}
    phases = sorted(set(ds.time_phases))
    for phase in phases:
        mask = np.array([p == phase for p in ds.time_phases])
        if mask.sum() == 0:
            continue
        pred = predict_with_weights(ds.X[mask], weights, intercept)
        acc = accuracy(pred, ds.y[mask])
        out[phase] = {"n": int(mask.sum()), "acc": acc}
    return out


# ============================
# main
# ============================


def evaluate_default_full(
    ds_full: Dataset, ds_reduced: Dataset, splits: list[Split],
) -> dict[str, dict[str, float]]:
    """DEFAULT_WEIGHTS の精度を full / reduced 両方で評価する。"""
    out: dict[str, dict[str, float]] = {}
    w_full = default_weights_vector()
    w_red = np.array([
        DEFAULT_WEIGHTS.get(n, 0.0) for n in ds_reduced.feature_names
    ])
    for split in splits:
        out[f"full__{split.label}"] = {
            "test_acc": accuracy(
                predict_with_weights(ds_full.X[split.test_idx], w_full),
                ds_full.y[split.test_idx],
            ),
        }
        out[f"reduced__{split.label}"] = {
            "test_acc": accuracy(
                predict_with_weights(ds_reduced.X[split.test_idx], w_red),
                ds_reduced.y[split.test_idx],
            ),
        }
    return out


def search_all_models(
    ds: Dataset, split: Split,
) -> list[ModelResult]:
    """全モデルを探索して結果リストを返す。"""
    results: list[ModelResult] = []
    for C in LR_L2_C_GRID:
        results.append(fit_lr_eval(ds, split, "l2", C))
    for C in LR_L1_C_GRID:
        results.append(fit_lr_eval(ds, split, "l1", C))
    for l1r in EN_L1_RATIO_GRID:
        for C in EN_C_GRID:
            results.append(fit_elastic_net(ds, split, l1r, C))
    for alpha in (0.1, 1.0, 5.0, 10.0):
        results.append(fit_ridge_classifier(ds, split, alpha))
    for d in RF_MAX_DEPTH_GRID:
        results.append(fit_rf_eval(ds, split, d))
    for alpha in (0.5, 1.0, 5.0):
        try:
            results.append(fit_sign_constrained_logreg(ds, split, alpha))
        except Exception as e:
            print(f"[warn] sign_constrained failed: {e}", file=sys.stderr)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="クリーン重み学習 v3")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--test-video", type=str, default="03")
    args = parser.parse_args()

    ds_full = load_dataset(args.csv)
    print(f"[load] full: n={len(ds_full.y)}, d={len(ds_full.feature_names)}")
    rd = reduce_features(ds_full)
    ds = reduced_to_dataset(rd)
    print(
        f"[reduce] dropped={list(rd.dropped)} -> "
        f"d={len(rd.feature_names)}",
    )

    # VIF 比較 (full vs reduced)
    vif_before = compute_vif(ds_full.X)
    vif_after = compute_vif(rd.X)
    print(
        f"[vif] before max={np.nanmax(vif_before):.2f}, "
        f"after max={np.nanmax(vif_after[np.isfinite(vif_after)]):.2f}",
    )

    splits = [
        video_level_split(ds, test_video=args.test_video),
        random_split(ds, train_ratio=0.7),
    ]
    full_splits = [
        video_level_split(ds_full, test_video=args.test_video),
        random_split(ds_full, train_ratio=0.7),
    ]

    default_acc = evaluate_default_full(ds_full, ds, full_splits)
    print(f"[default] {default_acc}")

    all_results: list[ModelResult] = []
    for split in splits:
        print(
            f"\n[split={split.label}] train={len(split.train_idx)}, "
            f"test={len(split.test_idx)}",
        )
        for r in search_all_models(ds, split):
            print(
                f"  {r.model} {r.params}: "
                f"train={r.train_acc:.3f}, test={r.test_acc:.3f}",
            )
            all_results.append(r)

    # video holdout で最良
    video_results = [
        r for r in all_results
        if "video_holdout" in r.split and r.weights
    ]
    best = max(video_results, key=lambda r: r.test_acc)
    print(
        f"\n[best v3 (video holdout)] {best.model} {best.params}: "
        f"test_acc={best.test_acc:.3f}",
    )

    # K-fold
    kfold = kfold_eval(ds, KFOLD_K, fit_lr_eval, penalty="l2", C=1.0)
    print(f"[kfold k={KFOLD_K}] mean={kfold['mean']:.3f} ± {kfold['std']:.3f}")

    # per-phase 精度 (best 重みを ds_full に embed して評価)
    weights_full = np.zeros(len(ds_full.feature_names))
    for i, n in enumerate(ds_full.feature_names):
        weights_full[i] = best.weights.get(n, 0.0)
    per_phase = per_phase_acc(ds_full, weights_full, best.intercept)

    # スコア表示用に DEFAULT スケールへ正規化
    best_weights_normalized = normalize_weights_to_default_scale(best.weights)
    # full 16 特徴量に展開 (削除特徴量は 0)
    v3_global_weights: dict[str, float] = {
        n: 0.0 for n in FEATURE_NAMES
    }
    for n, v in best_weights_normalized.items():
        v3_global_weights[n] = float(v)

    out: dict[str, Any] = {
        "n_samples": len(ds.y),
        "n_features_full": len(ds_full.feature_names),
        "n_features_reduced": len(rd.feature_names),
        "dropped_features": list(rd.dropped),
        "vif_before_max": float(np.nanmax(vif_before)),
        "vif_after_max": float(
            np.nanmax(vif_after[np.isfinite(vif_after)]),
        ),
        "vif_before": {
            n: (None if not np.isfinite(v) else float(v))
            for n, v in zip(ds_full.feature_names, vif_before)
        },
        "vif_after": {
            n: (None if not np.isfinite(v) else float(v))
            for n, v in zip(rd.feature_names, vif_after)
        },
        "default_baseline": default_acc,
        "per_model_results": [serialize_result(r) for r in all_results],
        "best_overall": serialize_result(best),
        "best_overall_normalized": best_weights_normalized,
        "best_overall_v3_global": v3_global_weights,
        "kfold": kfold,
        "per_phase_acc_with_best": per_phase,
        "physically_positive_features": list(PHYSICALLY_POSITIVE_FEATURES),
        "physically_negative_features": list(PHYSICALLY_NEGATIVE_FEATURES),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
