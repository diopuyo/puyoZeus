"""不採用の診断実験。配信経路に接続しないスカラー確率変換。"""
from __future__ import annotations
import math
from typing import Any
import numpy as np

EXP_OVERFLOW_BOUND = 709.


class ScalarLogitLink:
    """入力ロジットをスカラーexpで変換し、既存リンクの属性を保持する。"""

    def __init__(self, original: Any) -> None:
        self.original = original

    def __getattr__(self, key: str) -> Any:
        return getattr(self.original, key)

    def inverse(self, raw_prediction: Any, out: Any = None) -> Any:
        target = np.empty_like(raw_prediction) if out is None else out
        for index in np.ndindex(raw_prediction.shape):
            value = float(raw_prediction[index])
            target[index] = scalar_expit(value)
        return target


def configure(predictor: Any) -> None:
    """このプロセスでロードしたv4.9f専用モデルだけを変更する。"""
    for evaluator in predictor.evaluators:
        if evaluator is not None:
            loss = evaluator.model._loss
            loss.link = ScalarLogitLink(loss.link)
    from src import prefire_choice_model, prefire_post_v46
    prefire_choice_model.expit = scalar_expit
    prefire_post_v46.expit = scalar_expit


def scalar_expit(value: float) -> float:
    return 0. if value < -EXP_OVERFLOW_BOUND else 1. / (1. + math.exp(-float(value)))
