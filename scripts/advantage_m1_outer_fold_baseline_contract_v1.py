"""二段目の評価foldを見ていないM1 baselineだけを許可する純粋検査。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


FOLD_IDS = (1, 2, 3, 4, 5, 6)
M1_SEEDS = (20260904, 20260905, 20260906)


class OuterFoldBaselineError(ValueError):
    """baselineの外側fold、seed、または既存OOFとの同一性が不正。"""


def validate_outer_fold_record(record: Mapping[str, Any], outer_fold: int) -> None:
    """親dataset/SHA検証済みrecordの学習・調整由来だけを検査する。"""
    if type(outer_fold) is not int or outer_fold not in FOLD_IDS:
        raise OuterFoldBaselineError("outer foldは整数1〜6が必要です")
    tune = outer_fold % len(FOLD_IDS) + 1
    expected = tuple(value for value in FOLD_IDS if value not in (outer_fold, tune))
    train = record.get("train_folds")
    if not isinstance(train, (list, tuple)) or any(type(value) is not int for value in train):
        raise OuterFoldBaselineError("train foldsが整数列ではありません")
    if (type(record.get("eval_fold")) is not int or record["eval_fold"] != outer_fold
            or type(record.get("tune_fold")) is not int or record["tune_fold"] != tune
            or tuple(sorted(train)) != expected):
        raise OuterFoldBaselineError("M1 checkpointのtrain/tune/evalが外側foldと不一致です")


def validate_outer_fold_seed_records(
    records: Mapping[int, Mapping[str, Any]], outer_fold: int,
) -> None:
    """外側foldごとに同じ評価foldの固定3seedだけを要求する。"""
    if set(records) != set(M1_SEEDS) or any(type(seed) is not int for seed in records):
        raise OuterFoldBaselineError("固定3seedが過不足なく必要です")
    for seed in M1_SEEDS:
        validate_outer_fold_record(records[seed], outer_fold)


def _probability(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise OuterFoldBaselineError(f"{name}の1次元・非空・有限条件が不正です")
    if np.any((array < 0.0) | (array > 1.0)):
        raise OuterFoldBaselineError(f"{name}が0〜1範囲外です")
    return array


def equal_seed_probability_mean(values: Mapping[int, np.ndarray]) -> np.ndarray:
    """seedごとに較正した確率等を固定順・float64で平均する。"""
    if set(values) != set(M1_SEEDS) or any(type(seed) is not int for seed in values):
        raise OuterFoldBaselineError("平均対象には固定3seedだけが必要です")
    arrays = [_probability(values[seed], str(seed)) for seed in M1_SEEDS]
    if len({array.shape for array in arrays}) != 1:
        raise OuterFoldBaselineError("seed間で確率shapeが不一致です")
    return np.mean(np.stack(arrays), axis=0, dtype=np.float64)


def assert_outer_oof_exact(
    predicted: np.ndarray, stored: np.ndarray, candidate_folds: Sequence[int],
    outer_fold: int,
) -> int:
    """外側eval対象の保存OOFと完全一致を要求し、比較行数を返す。"""
    actual, expected = _probability(predicted, "predicted"), _probability(stored, "stored")
    folds = np.asarray(candidate_folds)
    if (type(outer_fold) is not int or outer_fold not in FOLD_IDS
            or folds.shape != actual.shape or expected.shape != actual.shape
            or folds.dtype.kind not in "iu"
            or not np.isin(folds, FOLD_IDS).all()):
        raise OuterFoldBaselineError("OOF比較のfold/shapeが不正です")
    mask = folds == outer_fold
    if not mask.any() or not np.array_equal(actual[mask], expected[mask]):
        raise OuterFoldBaselineError("外側evalの保存OOF予測と完全一致しません")
    return int(mask.sum())
