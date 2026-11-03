"""対象側の得点欄から旧ゼロ判定の連続スコアを蒸留する。"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.score_zero import SCORE_1P_REGION, SCORE_2P_REGION, ZERO_NCC_THRESHOLD

SCORE_REGIONS = (SCORE_1P_REGION, SCORE_2P_REGION)
MIN_PIXEL_NORM = 1.0
MIN_SCORE = -1.0
MAX_SCORE = 1.0


def zero_features(frame: np.ndarray, side: int) -> np.ndarray:
    """対象の得点欄だけを濃淡正規化し、盤面・他方の得点を混ぜない。"""
    y1, y2, x1, x2 = SCORE_REGIONS[side]
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    values = gray.astype(np.float32).ravel()
    values -= values.mean()
    return values / max(float(np.linalg.norm(values)), MIN_PIXEL_NORM)


class LocalZeroModel:
    """学習した回帰器のスコアに旧閾値を適用し、採否との矛盾を防ぐ。"""

    def __init__(self, regressors: tuple[Any, Any]) -> None:
        self.regressors = regressors

    def predict_image(self, frame: np.ndarray) -> dict[str, bool | float]:
        scores = [float(np.clip(model.predict(zero_features(frame, side)[None])[0],
                                MIN_SCORE, MAX_SCORE))
                  for side, model in enumerate(self.regressors)]
        return dict(is_1p_zero=scores[0] >= ZERO_NCC_THRESHOLD,
                    is_2p_zero=scores[1] >= ZERO_NCC_THRESHOLD,
                    score_1p=scores[0], score_2p=scores[1])
