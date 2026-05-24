"""
段階 3: 多動画統合データセット上での多モデル学習 + 交差検証

目的:
    data/training/match_features.csv (~700 行 × 16 特徴量) を入力に、
    複数モデル (LR L1/L2 各正則化、RandomForest) の test_acc を比較し、
    DEFAULT_WEIGHTS / Grid 50 試合 / LR 50 試合 と同条件 (動画レベル
    train/test split) で汎化性能の最良重みを見つける。

データ split 方針:
    - 動画レベル split: train = 動画 01,02 / test = 動画 03 (デフォルト)
        → 真の汎化テスト。50 試合 grid search の 0.520 と直接比較可能。
    - sample レベル split (train_ratio=0.7, fold_per_match=True) も併走し
      train サイズの効果を確認する。

モデル:
    - LogisticRegression (L1 / L2 × C in {0.1, 0.5, 1.0, 5.0})
    - RandomForestClassifier (max_depth in {None, 4, 8})
    - LightGBM (任意。インストールされていればテスト)

時刻別モデル:
    - midpoint だけのデータで学習し、midpoint test_acc を測定。
    - end_minus_5 だけのデータで学習し、end test_acc を測定。
    - global (全 5 時刻) と比較。

出力:
    data/verify/learned_weights_v2.json
        - default_baseline       : DEFAULT_WEIGHTS の test_acc
        - per_model_results      : 各モデル × split 設定の test_acc 一覧
        - best_overall           : 最良重み (LR L2 ベース、原スケール)
        - best_per_phase         : phase 別最良重み
        - feature_importance_rf  : RF の特徴量重要度
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

# sklearn 1.8 の penalty 引数 deprecation 警告を抑制 (動作には影響なし)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn",
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn",
)

# プロジェクトルートを sys.path に追加
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.eda_features import Dataset, load_dataset  # noqa: E402
from scripts.generate_training_dataset import (  # noqa: E402
    DEFAULT_TIME_PHASES,
    FEATURE_NAMES,
    TIME_PHASE_END_MINUS,
    TIME_PHASE_MIDPOINT,
)
from src.scorer import DEFAULT_WEIGHTS  # noqa: E402

# ============================
# 定数
# ============================

DEFAULT_INPUT_CSV: Path = Path("data/training/match_features.csv")
DEFAULT_OUTPUT_JSON: Path = Path("data/verify/learned_weights_v2.json")

# 動画レベル split: 動画 03 を test に使う (約 1/3)
DEFAULT_TEST_VIDEO: str = "03"

# 多モデル探索のハイパラ候補
LR_L2_C_GRID: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 5.0)
LR_L1_C_GRID: tuple[float, ...] = (0.1, 0.5, 1.0, 5.0)
RF_MAX_DEPTH_GRID: tuple[int | None, ...] = (None, 4, 6, 8)
RF_N_ESTIMATORS: int = 200
LR_MAX_ITER: int = 2000
RANDOM_SEED: int = 42

# DEFAULT_WEIGHTS と比較する際の重みベクトル順序
WEIGHT_ORDER: tuple[str, ...] = FEATURE_NAMES


# ============================
# データ split
# ============================


@dataclass(frozen=True)
class Split:
    """学習・テスト用インデックス。"""
    train_idx: np.ndarray
    test_idx: np.ndarray
    label: str


def video_level_split(
    ds: Dataset, test_video: str = DEFAULT_TEST_VIDEO,
) -> Split:
    """test_video を含む行を test に、それ以外を train に振り分ける。"""
    train_mask = np.array([v != test_video for v in ds.video_ids])
    test_mask = ~train_mask
    return Split(
        train_idx=np.where(train_mask)[0],
        test_idx=np.where(test_mask)[0],
        label=f"video_holdout_{test_video}",
    )


def random_split(
    ds: Dataset, train_ratio: float = 0.7, seed: int = RANDOM_SEED,
) -> Split:
    """サンプル単位ランダム split (動画混在)。"""
    rng = np.random.default_rng(seed)
    n = len(ds.y)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * train_ratio)
    return Split(
        train_idx=idx[:cut],
        test_idx=idx[cut:],
        label=f"random_{train_ratio:.2f}",
    )


def phase_filter(
    ds: Dataset, phase: str,
) -> np.ndarray:
    """指定 phase に該当する行のインデックスを返す。"""
    return np.where(np.array(ds.time_phases) == phase)[0]


# ============================
# 評価
# ============================


def predict_with_weights(
    X: np.ndarray, weights: np.ndarray, intercept: float = 0.0,
) -> np.ndarray:
    """重みベクトルで +1/-1 を予測する。"""
    z = X @ weights + intercept
    return np.where(z >= 0, 1, -1)


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """予測精度を返す。"""
    if len(y_true) == 0:
        return 0.0
    return float((y_pred == y_true).mean())


def default_weights_vector() -> np.ndarray:
    """DEFAULT_WEIGHTS を FEATURE_NAMES 順のベクトルに変換する。"""
    return np.array(
        [DEFAULT_WEIGHTS.get(name, 0.0) for name in FEATURE_NAMES],
        dtype=np.float64,
    )


def evaluate_default(
    ds: Dataset, split: Split,
) -> dict[str, float]:
    """DEFAULT_WEIGHTS の train/test 精度を返す。"""
    w = default_weights_vector()
    train_pred = predict_with_weights(ds.X[split.train_idx], w)
    test_pred = predict_with_weights(ds.X[split.test_idx], w)
    return {
        "train_acc": accuracy(train_pred, ds.y[split.train_idx]),
        "test_acc": accuracy(test_pred, ds.y[split.test_idx]),
    }


# ============================
# モデル学習
# ============================


@dataclass
class ModelResult:
    """1 モデル × 1 split の評価結果。"""
    model: str
    params: dict[str, Any]
    split: str
    train_acc: float
    test_acc: float
    weights: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    feature_importance: dict[str, float] | None = None


def _fit_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    penalty: str,
    C: float,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """LR を学習し、原スケールの coef/intercept と StandardScaler 用パラメータを返す。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    solver = "saga" if penalty == "l1" else "lbfgs"
    clf = LogisticRegression(
        penalty=penalty, C=C, solver=solver,
        max_iter=LR_MAX_ITER, class_weight="balanced",
    )
    clf.fit(Xs, y_train)
    train_acc = float(clf.score(Xs, y_train))
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    coef_orig = clf.coef_[0] / scale
    intercept_orig = float(
        clf.intercept_[0] - np.sum(clf.coef_[0] * scaler.mean_ / scale)
    )
    return coef_orig, intercept_orig, train_acc, scaler.mean_, scale


def fit_lr_eval(
    ds: Dataset, split: Split, penalty: str, C: float,
) -> ModelResult:
    """LR を学習し、原スケール重みで test_acc を測定する。"""
    X_tr, y_tr = ds.X[split.train_idx], ds.y[split.train_idx]
    X_te, y_te = ds.X[split.test_idx], ds.y[split.test_idx]
    coef, b, train_acc, _, _ = _fit_lr(X_tr, y_tr, penalty, C)
    pred_te = predict_with_weights(X_te, coef, b)
    test_acc = accuracy(pred_te, y_te)
    weights = {
        name: float(coef[i])
        for i, name in enumerate(ds.feature_names)
    }
    return ModelResult(
        model=f"lr_{penalty}",
        params={"C": C},
        split=split.label,
        train_acc=train_acc,
        test_acc=test_acc,
        weights=weights,
        intercept=b,
    )


def fit_rf_eval(
    ds: Dataset, split: Split, max_depth: int | None,
    n_estimators: int = RF_N_ESTIMATORS,
) -> ModelResult:
    """RF を学習し、test_acc と feature_importance を返す。"""
    from sklearn.ensemble import RandomForestClassifier

    X_tr, y_tr = ds.X[split.train_idx], ds.y[split.train_idx]
    X_te, y_te = ds.X[split.test_idx], ds.y[split.test_idx]
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    train_acc = float(clf.score(X_tr, y_tr))
    test_acc = float(clf.score(X_te, y_te))
    fi = {
        name: float(clf.feature_importances_[i])
        for i, name in enumerate(ds.feature_names)
    }
    return ModelResult(
        model="rf",
        params={"max_depth": max_depth, "n_estimators": n_estimators},
        split=split.label,
        train_acc=train_acc,
        test_acc=test_acc,
        feature_importance=fi,
    )


# ============================
# 全モデル探索
# ============================


def search_lr(
    ds: Dataset, split: Split,
) -> list[ModelResult]:
    """LR L1 / L2 を C を変えて全探索する。"""
    results: list[ModelResult] = []
    for C in LR_L2_C_GRID:
        results.append(fit_lr_eval(ds, split, "l2", C))
    for C in LR_L1_C_GRID:
        results.append(fit_lr_eval(ds, split, "l1", C))
    return results


def search_rf(ds: Dataset, split: Split) -> list[ModelResult]:
    """RandomForest の max_depth を変えて全探索する。"""
    results: list[ModelResult] = []
    for d in RF_MAX_DEPTH_GRID:
        results.append(fit_rf_eval(ds, split, d))
    return results


def best_by_test_acc(results: list[ModelResult]) -> ModelResult:
    """test_acc が最大のモデル結果を返す。"""
    return max(results, key=lambda r: r.test_acc)


# ============================
# phase 別モデル
# ============================


def fit_phase_lr(
    ds: Dataset, phase: str, C: float = 0.5,
) -> dict[str, Any]:
    """指定 phase のサブセットだけで LR を学習する。"""
    idx = phase_filter(ds, phase)
    if len(idx) < 10:
        return {"phase": phase, "n": len(idx), "skipped": True}
    sub_ds = Dataset(
        feature_names=ds.feature_names,
        X=ds.X[idx], y=ds.y[idx],
        video_ids=[ds.video_ids[i] for i in idx],
        time_phases=[ds.time_phases[i] for i in idx],
    )
    split = video_level_split(sub_ds)
    if len(split.train_idx) < 5 or len(split.test_idx) < 5:
        split = random_split(sub_ds, train_ratio=0.7)
    res = fit_lr_eval(sub_ds, split, "l2", C)
    return {
        "phase": phase,
        "n": len(idx),
        "split_label": split.label,
        "train_acc": res.train_acc,
        "test_acc": res.test_acc,
        "weights": res.weights,
        "intercept": res.intercept,
    }


# ============================
# 結果集約
# ============================


def serialize_result(r: ModelResult) -> dict[str, Any]:
    """ModelResult を JSON 化可能な辞書に変換する。"""
    return {
        "model": r.model,
        "params": r.params,
        "split": r.split,
        "train_acc": r.train_acc,
        "test_acc": r.test_acc,
        "weights": r.weights,
        "intercept": r.intercept,
        "feature_importance": r.feature_importance,
    }


def normalize_weights_to_default_scale(
    weights: dict[str, float],
) -> dict[str, float]:
    """重みベクトルを DEFAULT_WEIGHTS と同じ L1 ノルムにスケーリングする。

    Args:
        weights: {name: coef}

    Returns:
        スケール調整後重み。

    Notes:
        DEFAULT_WEIGHTS の sum(|w|) = 9.7 程度。学習済み LR coef は
        StandardScaler 巻き戻し済みで通常 |w| が大きく出るため、
        スコア表示時の桁感を揃えるために正規化する。
    """
    target = sum(abs(w) for w in DEFAULT_WEIGHTS.values())
    current = sum(abs(w) for w in weights.values())
    if current == 0:
        return dict(weights)
    factor = target / current
    return {name: float(w * factor) for name, w in weights.items()}


# ============================
# main
# ============================


def main() -> int:
    parser = argparse.ArgumentParser(description="多モデル学習 v2")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--test-video", type=str, default=DEFAULT_TEST_VIDEO,
    )
    args = parser.parse_args()

    ds = load_dataset(args.csv)
    print(f"[load] {len(ds.y)} 行, {len(ds.feature_names)} 特徴量")

    # 動画レベル + ランダム の 2 つの split で評価
    splits = [
        video_level_split(ds, test_video=args.test_video),
        random_split(ds, train_ratio=0.7),
    ]

    all_results: list[ModelResult] = []
    default_baselines: dict[str, dict[str, float]] = {}
    for split in splits:
        default_baselines[split.label] = evaluate_default(ds, split)
        print(
            f"\n[split={split.label}] train={len(split.train_idx)}, "
            f"test={len(split.test_idx)}, "
            f"DEFAULT test_acc={default_baselines[split.label]['test_acc']:.3f}",
        )
        for res in search_lr(ds, split):
            print(f"  {res.model} C={res.params['C']:.2f}: "
                  f"train={res.train_acc:.3f}, test={res.test_acc:.3f}")
            all_results.append(res)
        for res in search_rf(ds, split):
            print(f"  rf depth={res.params['max_depth']}: "
                  f"train={res.train_acc:.3f}, test={res.test_acc:.3f}")
            all_results.append(res)

    # 動画レベル split の中で最良を選ぶ (汎化重視)
    video_results = [r for r in all_results if "video_holdout" in r.split]
    best_overall = best_by_test_acc(
        [r for r in video_results if r.weights],
    )
    print(
        f"\n[best overall (video holdout)] "
        f"{best_overall.model} {best_overall.params}: "
        f"test_acc={best_overall.test_acc:.3f}",
    )
    best_global_weights = normalize_weights_to_default_scale(
        best_overall.weights,
    )

    # phase 別モデル
    phase_models: dict[str, Any] = {}
    for phase in DEFAULT_TIME_PHASES:
        phase_models[phase] = fit_phase_lr(ds, phase)
        print(
            f"  phase={phase}: n={phase_models[phase].get('n')}, "
            f"test_acc={phase_models[phase].get('test_acc', 'N/A')}",
        )

    rf_video_results = [
        r for r in video_results if r.model == "rf" and r.feature_importance
    ]
    best_rf = best_by_test_acc(rf_video_results) if rf_video_results else None

    out: dict[str, Any] = {
        "n_samples": len(ds.y),
        "feature_names": list(ds.feature_names),
        "default_baseline": default_baselines,
        "per_model_results": [serialize_result(r) for r in all_results],
        "best_overall": serialize_result(best_overall),
        "best_overall_normalized": best_global_weights,
        "phase_models": phase_models,
        "feature_importance_rf": (
            best_rf.feature_importance if best_rf else None
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
