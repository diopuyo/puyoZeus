"""同一更新の候補をまとめ、固定モデルの列順とfloat32丸めを保つ。"""
from __future__ import annotations

from typing import Any

import numpy as np


def predict_rows(model: Any, rows: list[dict]) -> list[float]:
    from src.b3_ensemble import EnsembleTwoStateModel, aggregate_probabilities
    delegated = getattr(model, 'predict_light_rows', None)
    if callable(delegated):
        return delegated(rows)
    if type(model) is not EnsembleTwoStateModel:
        return [model.predict_row(row) for row in rows]
    stack = np.asarray([member_rows(member, rows) for member in model.members])
    return aggregate_probabilities(stack, model.aggregate).tolist()


def member_rows(member: Any, rows: list[dict]) -> np.ndarray:
    from src.b3_predict import RowPredictor
    from src.b3_train_model import MODE_COLUMN, THREADS, hgb_threads
    predictor = getattr(member, '_row_predictor', None)
    if predictor is None:
        predictor = member._row_predictor = RowPredictor(member)
    modes = np.asarray([float(row[MODE_COLUMN]) for row in rows])
    if np.isnan(modes).any():
        raise ValueError('mode_exchange に欠測 (状態を選べない)')
    result = np.full(len(rows), np.nan)
    for mode in (0, 1):
        indices = np.flatnonzero(modes == mode)
        if not len(indices):
            continue
        evaluator, getter = predictor.evaluators[mode], predictor.getters[mode]
        if evaluator is None:
            raise ValueError(f'mode={mode} の行があるが評価器が無い')
        matrix = np.asarray([getter(rows[i]) for i in indices], dtype=np.float32)
        with predictor.pools.limit(limits=THREADS, user_api='blas'), \
                predictor.pools.limit(limits=hgb_threads(), user_api='openmp'):
            result[indices] = evaluator.model.predict_proba(matrix)[:, 1]
    return result
