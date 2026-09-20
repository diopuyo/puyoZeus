"""固定6分割で暫定モデルのリークなし予測を作る共通処理。"""

from __future__ import annotations

import hashlib
import math
import pickle
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FOLD_IDS = (1, 2, 3, 4, 5, 6)
PROVISIONAL_RANDOM_STATE = 20260830
MODEL_FAMILIES = ("linear", "tree")
PREDICTION_CONTRACT_VERSION = "symmetric-average-v1"
LINEAR_REGULARIZATION_C = 0.1
PROBABILITY_EPSILON = 1e-6
CALIBRATION_BIN_COUNT = 10
MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 120,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "early_stopping": False,
}


class EventProvisionalOofError(ValueError):
    """固定分割・左右対称・学習重みの条件を満たさない。"""


@dataclass(frozen=True, slots=True)
class FoldPlan:
    """一回の評価に使う学習・調整・評価分割。"""

    eval_fold: int
    tune_fold: int
    train_folds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PlattCalibration:
    """生確率のlogitへ適用する一次変換。"""

    slope: float
    intercept: float


@dataclass(frozen=True, slots=True)
class FoldModelRecord:
    """一評価分割のモデルと検収値。"""

    plan: FoldPlan
    model_family: str
    model_hash: str
    calibration: PlattCalibration
    metrics_raw: dict[str, float]
    metrics_calibrated: dict[str, float]
    train_row_count: int
    tune_row_count: int
    eval_row_count: int


@dataclass(frozen=True, slots=True)
class FixedOofResult:
    """全行と対応する予測、および6個の分割モデル。"""

    raw_probability: np.ndarray
    calibrated_probability: np.ndarray
    model_hash_by_row: np.ndarray
    records: tuple[FoldModelRecord, ...]
    models: tuple[Any, ...]


def fixed_fold_plan(folds: Sequence[int]) -> tuple[FoldPlan, ...]:
    """評価k、調整k+1、残り4分割を学習とする順序を固定する。"""

    observed = {int(value) for value in folds}
    if observed != set(FOLD_IDS):
        raise EventProvisionalOofError(f"分割は1〜6が全て必要です: {sorted(observed)}")
    plans = []
    for eval_fold in FOLD_IDS:
        tune_fold = eval_fold % len(FOLD_IDS) + 1
        train = tuple(fold for fold in FOLD_IDS if fold not in {eval_fold, tune_fold})
        plans.append(FoldPlan(eval_fold, tune_fold, train))
    return tuple(plans)


def mirror_feature_matrix(
    matrix: np.ndarray, feature_names: Sequence[str],
) -> np.ndarray:
    """1P/2P列を交換し、差分だけ符号反転した入力を返す。"""

    source = np.asarray(matrix, dtype=float)
    if source.ndim != 2 or source.shape[1] != len(feature_names):
        raise EventProvisionalOofError("入力行列と列名の形が一致しません")
    positions = {name: index for index, name in enumerate(feature_names)}
    if len(positions) != len(feature_names):
        raise EventProvisionalOofError("入力列名が重複しています")
    mirrored = np.empty_like(source)
    for index, name in enumerate(feature_names):
        counterpart, negate = _mirror_rule(name)
        source_index = positions.get(counterpart) if counterpart else index
        if source_index is None:
            raise EventProvisionalOofError(f"左右対応列がありません: {name}")
        mirrored[:, index] = source[:, source_index]
        if negate:
            mirrored[:, index] *= -1.0
    return mirrored


def _mirror_rule(name: str) -> tuple[str | None, bool]:
    if name.endswith("_p1_missing"):
        return name.removesuffix("_p1_missing") + "_p2_missing", False
    if name.endswith("_p2_missing"):
        return name.removesuffix("_p2_missing") + "_p1_missing", False
    if name.endswith("_p1_norm"):
        return name.removesuffix("_p1_norm") + "_p2_norm", False
    if name.endswith("_p2_norm"):
        return name.removesuffix("_p2_norm") + "_p1_norm", False
    if name.endswith("_p1"):
        return name.removesuffix("_p1") + "_p2", False
    if name.endswith("_p2"):
        return name.removesuffix("_p2") + "_p1", False
    if name.endswith("_diff"):
        return None, True
    return None, False


def augment_training_with_mirror(
    matrix: np.ndarray, labels: np.ndarray, weights: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """左右反転標本を足し、元の総weightを半分ずつに保つ。"""

    x = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=int)
    w = np.asarray(weights, dtype=float)
    if len(x) != len(y) or len(y) != len(w):
        raise EventProvisionalOofError("左右増強する行数が一致しません")
    if np.any((y != 0) & (y != 1)) or np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise EventProvisionalOofError("勝敗または学習重みが不正です")
    return (
        np.concatenate([x, mirror_feature_matrix(x, feature_names)]),
        np.concatenate([y, 1 - y]),
        np.concatenate([w / 2.0, w / 2.0]),
    )


def symmetric_predict_probability(
    model: Any, matrix: np.ndarray, feature_names: Sequence[str],
) -> np.ndarray:
    """元入力と左右交換入力を平均し、1P/2P交換で確率を厳密反転する。"""

    source = np.asarray(matrix, dtype=float)
    original = model.predict_proba(source)[:, 1]
    mirrored = model.predict_proba(
        mirror_feature_matrix(source, feature_names),
    )[:, 1]
    return np.clip(0.5 * (original + 1.0 - mirrored), 0.0, 1.0)


def fit_platt_calibration(
    raw_probability: np.ndarray, labels: np.ndarray, weights: np.ndarray,
) -> PlattCalibration:
    """調整分割だけで左右対称なPlatt変換を学習する。"""

    raw = _clip_probability(raw_probability)
    y = np.asarray(labels, dtype=int)
    w = np.asarray(weights, dtype=float)
    if len(raw) != len(y) or len(y) != len(w) or len(raw) == 0:
        raise EventProvisionalOofError("確率調整の行数が不正です")
    logits = _logit(raw)
    x = np.concatenate([logits, -logits]).reshape(-1, 1)
    symmetric_y = np.concatenate([y, 1 - y])
    symmetric_w = np.concatenate([w / 2.0, w / 2.0])
    model = LogisticRegression(
        C=1_000_000.0, solver="lbfgs", max_iter=1000, fit_intercept=False,
    )
    model.fit(x, symmetric_y, sample_weight=symmetric_w)
    slope = float(model.coef_[0, 0])
    if not math.isfinite(slope) or slope < 0.0:
        raise EventProvisionalOofError(
            f"Platt較正のslopeが単調非減少ではありません: {slope}",
        )
    return PlattCalibration(slope, 0.0)


def apply_platt_calibration(
    raw_probability: np.ndarray, params: PlattCalibration,
) -> np.ndarray:
    """保存済みPlatt係数を確率配列へ適用する。"""

    transformed = params.slope * _logit(_clip_probability(raw_probability)) + params.intercept
    transformed = np.clip(transformed, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-transformed))


def weighted_probability_metrics(
    labels: np.ndarray, probability: np.ndarray, weights: np.ndarray,
) -> dict[str, float]:
    """試合均等weightで確率精度と重大な逆方向を測る。"""

    y = np.asarray(labels, dtype=int)
    p = _clip_probability(probability)
    w = np.asarray(weights, dtype=float)
    if len(y) != len(p) or len(p) != len(w) or len(y) == 0:
        raise EventProvisionalOofError("確率評価の行数が不正です")
    severe = ((y == 1) & (p <= 0.1)) | ((y == 0) & (p >= 0.9))
    auc = float("nan") if len(np.unique(y)) < 2 else float(
        roc_auc_score(y, p, sample_weight=w)
    )
    calibration = _weighted_calibration_metrics(y, p, w)
    return {
        "log_loss": float(log_loss(y, p, sample_weight=w, labels=[0, 1])),
        "brier": float(np.average((p - y) ** 2, weights=w)),
        "auc": auc,
        "severe_wrong_rate": float(np.average(severe.astype(float), weights=w)),
        "extreme_probability_rate": float(np.average(
            ((p <= 0.05) | (p >= 0.95)).astype(float), weights=w,
        )),
        **calibration,
        "weight_sum": float(w.sum()),
    }


def _weighted_calibration_metrics(
    labels: np.ndarray, probability: np.ndarray, weights: np.ndarray,
) -> dict[str, float]:
    """固定10区分で表示確率と実勝率の重み付きずれを返す。"""

    bin_index = np.minimum(
        (probability * CALIBRATION_BIN_COUNT).astype(int),
        CALIBRATION_BIN_COUNT - 1,
    )
    total_weight = float(weights.sum())
    weighted_gap = 0.0
    maximum_gap = 0.0
    populated = 0
    for index in range(CALIBRATION_BIN_COUNT):
        mask = bin_index == index
        bin_weight = float(weights[mask].sum())
        if bin_weight <= 0:
            continue
        gap = abs(
            float(np.average(probability[mask], weights=weights[mask]))
            - float(np.average(labels[mask], weights=weights[mask]))
        )
        weighted_gap += bin_weight * gap
        maximum_gap = max(maximum_gap, gap)
        populated += 1
    return {
        "calibration_ece_10bin": weighted_gap / total_weight,
        "calibration_max_gap_10bin": maximum_gap,
        "calibration_populated_bins": float(populated),
    }


def linear_top_contributions(
    matrix: np.ndarray, folds: np.ndarray, prediction_mask: np.ndarray,
    result: FixedOofResult, feature_names: Sequence[str], *, top_k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """線形OOFモデルの各行で絶対寄与が大きい入力番号と符号付き値を返す。"""

    x = np.asarray(matrix, dtype=float)
    f = np.asarray(folds, dtype=int)
    predictable = np.asarray(prediction_mask, dtype=bool)
    if x.shape != (len(f), len(feature_names)) or len(predictable) != len(f):
        raise EventProvisionalOofError("寄与計算の入力形が一致しません")
    if top_k < 1 or top_k > len(feature_names):
        raise EventProvisionalOofError("寄与表示数が特徴量数の範囲外です")
    indices = np.full((len(x), top_k), -1, dtype=int)
    values = np.full((len(x), top_k), np.nan)
    for record, model in zip(result.records, result.models, strict=True):
        if record.model_family != "linear":
            raise EventProvisionalOofError("線形以外のモデルは係数寄与を出せません")
        mask = predictable & (f == record.plan.eval_fold)
        transformed = _transform_linear_input(model, x[mask])
        mirrored = _transform_linear_input(
            model, mirror_feature_matrix(x[mask], feature_names),
        )
        contributions = 0.5 * (transformed - mirrored)
        contributions *= model.named_steps["classifier"].coef_[0]
        order = np.argsort(-np.abs(contributions), axis=1)[:, :top_k]
        indices[mask] = order
        values[mask] = np.take_along_axis(contributions, order, axis=1)
    return indices, values


def _transform_linear_input(model: Any, matrix: np.ndarray) -> np.ndarray:
    values = model.named_steps["imputer"].transform(matrix)
    return model.named_steps["scaler"].transform(values)


def run_fixed_oof(
    matrix: np.ndarray, labels: np.ndarray, weights: np.ndarray,
    groups: np.ndarray, folds: np.ndarray, feature_names: Sequence[str],
    training_mask: np.ndarray, prediction_mask: np.ndarray,
    *, random_state: int = PROVISIONAL_RANDOM_STATE, model_family: str = "tree",
    monotonic_constraints: Mapping[str, int] | None = None,
) -> FixedOofResult:
    """固定6分割で学習・調整・評価を分離し全対象行のOOFを返す。"""

    x = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=int)
    w = np.asarray(weights, dtype=float)
    g = np.asarray(groups, dtype=str)
    f = np.asarray(folds, dtype=int)
    trainable = np.asarray(training_mask, dtype=bool)
    predictable = np.asarray(prediction_mask, dtype=bool)
    _validate_oof_arrays(x, y, w, g, f, trainable, predictable, feature_names)
    if model_family not in MODEL_FAMILIES:
        raise EventProvisionalOofError(f"未登録のモデル方式です: {model_family}")
    monotonic_cst = _monotonic_constraint_vector(
        feature_names, monotonic_constraints, model_family,
    )
    raw = np.full(len(x), np.nan)
    calibrated = np.full(len(x), np.nan)
    hashes = np.full(len(x), "", dtype=object)
    records: list[FoldModelRecord] = []
    models: list[Any] = []
    for plan in fixed_fold_plan(f.tolist()):
        record, model = _run_one_fold(
            x, y, w, f, feature_names, trainable, predictable,
            plan, raw, calibrated, hashes, random_state, model_family,
            monotonic_cst,
        )
        records.append(record)
        models.append(model)
    return FixedOofResult(raw, calibrated, hashes, tuple(records), tuple(models))


def _run_one_fold(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, folds: np.ndarray,
    names: Sequence[str], trainable: np.ndarray, predictable: np.ndarray,
    plan: FoldPlan, raw: np.ndarray, calibrated: np.ndarray,
    hashes: np.ndarray, random_state: int, model_family: str,
    monotonic_cst: tuple[int, ...] | None,
) -> tuple[FoldModelRecord, Any]:
    train_index = trainable & np.isin(folds, plan.train_folds)
    tune_index = trainable & (folds == plan.tune_fold)
    eval_index = trainable & (folds == plan.eval_fold)
    predict_index = predictable & (folds == plan.eval_fold)
    x_train, y_train, w_train = augment_training_with_mirror(
        x[train_index], y[train_index], w[train_index], names,
    )
    model = _fit_estimator(
        x_train, y_train, w_train, random_state, model_family, monotonic_cst,
    )
    tune_raw = symmetric_predict_probability(model, x[tune_index], names)
    calibration = fit_platt_calibration(tune_raw, y[tune_index], w[tune_index])
    raw[predict_index] = symmetric_predict_probability(
        model, x[predict_index], names,
    )
    calibrated[predict_index] = apply_platt_calibration(raw[predict_index], calibration)
    model_hash = _model_hash(model, calibration, names, plan)
    hashes[predict_index] = model_hash
    record = _fold_record(
        plan, model_family, model_hash, calibration, y, w, eval_index, raw, calibrated,
        int(train_index.sum()), int(tune_index.sum()),
    )
    return record, model


def _fit_estimator(
    x: np.ndarray, y: np.ndarray, w: np.ndarray,
    random_state: int, model_family: str,
    monotonic_cst: tuple[int, ...] | None = None,
) -> Any:
    if model_family == "tree":
        model = HistGradientBoostingClassifier(
            random_state=random_state, monotonic_cst=monotonic_cst, **MODEL_PARAMS,
        )
        model.fit(x, y, sample_weight=w)
        return model
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=LINEAR_REGULARIZATION_C, solver="lbfgs", max_iter=1000,
            random_state=random_state,
        )),
    ])
    model.fit(x, y, classifier__sample_weight=w)
    return model


def _monotonic_constraint_vector(
    feature_names: Sequence[str], constraints: Mapping[str, int] | None,
    model_family: str,
) -> tuple[int, ...] | None:
    """特徴量名で指定した単調制約を学習器の列順へ安全にそろえる。"""

    if not constraints:
        return None
    unknown = set(constraints) - set(feature_names)
    invalid = {name: value for name, value in constraints.items() if value not in {-1, 0, 1}}
    if unknown:
        raise EventProvisionalOofError(f"単調制約の未登録列があります: {sorted(unknown)}")
    if invalid:
        raise EventProvisionalOofError(f"単調制約は-1/0/1だけです: {invalid}")
    if model_family != "tree":
        raise EventProvisionalOofError("単調制約はtree方式だけで利用できます")
    return tuple(int(constraints.get(name, 0)) for name in feature_names)


def _fold_record(
    plan: FoldPlan, model_family: str, model_hash: str,
    calibration: PlattCalibration,
    y: np.ndarray, w: np.ndarray, eval_index: np.ndarray,
    raw: np.ndarray, calibrated: np.ndarray, train_count: int, tune_count: int,
) -> FoldModelRecord:
    return FoldModelRecord(
        plan, model_family, model_hash, calibration,
        weighted_probability_metrics(y[eval_index], raw[eval_index], w[eval_index]),
        weighted_probability_metrics(
            y[eval_index], calibrated[eval_index], w[eval_index],
        ),
        train_count, tune_count, int(eval_index.sum()),
    )


def _validate_oof_arrays(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, groups: np.ndarray,
    folds: np.ndarray, trainable: np.ndarray, predictable: np.ndarray,
    names: Sequence[str],
) -> None:
    lengths = {len(x), len(y), len(w), len(groups), len(folds), len(trainable), len(predictable)}
    if len(lengths) != 1 or x.ndim != 2 or x.shape[1] != len(names):
        raise EventProvisionalOofError("OOF入力配列の形が一致しません")
    if np.any(trainable & (~np.isfinite(w) | (w <= 0))):
        raise EventProvisionalOofError("学習対象のweightが不正です")
    if np.any(trainable & ((y != 0) & (y != 1))):
        raise EventProvisionalOofError("学習対象の勝敗が0/1ではありません")
    group_folds: dict[str, set[int]] = {}
    for group, fold in zip(groups, folds, strict=True):
        group_folds.setdefault(str(group), set()).add(int(fold))
    if any(len(values) != 1 for values in group_folds.values()):
        raise EventProvisionalOofError("同じ元映像群が複数foldへ漏れています")
    fixed_fold_plan(folds.tolist())


def _model_hash(
    model: Any, calibration: PlattCalibration,
    names: Sequence[str], plan: FoldPlan,
) -> str:
    payload = pickle.dumps((
        PREDICTION_CONTRACT_VERSION, model, calibration, tuple(names), plan,
    ), protocol=5)
    return hashlib.sha256(payload).hexdigest()


def _clip_probability(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise EventProvisionalOofError("確率に非有限値があります")
    return np.clip(array, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


def _logit(probability: np.ndarray) -> np.ndarray:
    return np.log(probability / (1.0 - probability))
