"""窒息位置の×印を画像非同梱の小型分類器で判定する未採用経路。"""
from __future__ import annotations

import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src.image_free_digit import digit_features
from src.ui_mask import UiMaskMatcher, UiMatchResult, DEFAULT_NCC_THRESHOLD

MODEL_SCHEMA = 1
MODEL_ENV = 'PUYO_IMAGE_FREE_UI_MASK_MODEL'
DEFAULT_MODEL = Path('models/image_free/ui_mask.joblib')
CONTEXT_MIN_CONFIDENCE = 0.65
CONTEXT_MIN_CORRELATION = 0.5
CROSS_ARM_WIDTH = 0.5
CROSS_ARM_LENGTH = 1.35
COLOR_VARIATION_MIN = 1.0
CIRCLE_RADIUS = 0.9
CIRCLE_CORRELATION_MARGIN = 0.02
PATCH_MEMO_SIZE = 16    # 同じ画素のセルの判定を使い回す件数 (フィールドの読みの 2 か所から同じセルを見るため、11/14)


@lru_cache(maxsize=8)
def _cross_geometry(height: int, width: int) -> np.ndarray:
    """保存画像を使わず、二本の斜線の交差を数式で表す。"""
    y, x = np.mgrid[:height, :width].astype(float)
    x = (x - (width - 1) / 2) / (width / 2)
    y = (y - (height - 1) / 2) / (height / 2)
    cross = ((np.abs(np.abs(x) - np.abs(y)) < CROSS_ARM_WIDTH)
             & (np.abs(x) + np.abs(y) < CROSS_ARM_LENGTH)).astype(float)
    cross.setflags(write=False)
    return cross


def _context_cross_score(context: np.ndarray) -> float:
    """赤と緑の差で黄色い発光の共通成分を抑え、外形を確認する。"""
    if context.size == 0 or context.ndim != 3:
        return 0.0
    color = context[:, :, 2].astype(float) - context[:, :, 1].astype(float)
    if color.std() < COLOR_VARIATION_MIN:
        return 0.0
    shape = _cross_geometry(*color.shape)
    cross_score = float(np.corrcoef(shape.ravel(), color.ravel())[0, 1])
    height, width = color.shape
    y, x = np.mgrid[:height, :width].astype(float)
    x = (x - (width - 1) / 2) / (width / 2)
    y = (y - (height - 1) / 2) / (height / 2)
    circle = (x*x + y*y < CIRCLE_RADIUS*CIRCLE_RADIUS).astype(float)
    circle_score = float(np.corrcoef(circle.ravel(), color.ravel())[0, 1])
    if circle_score > cross_score + CIRCLE_CORRELATION_MARGIN:
        return 0.0
    return cross_score


class ImageFreeUiMaskMatcher(UiMaskMatcher):
    """画像へのフォールバックを持たず、既定の採否を学習した分類器。"""

    def __init__(self, model_path: Path | None = None,
                 threshold: float = DEFAULT_NCC_THRESHOLD) -> None:
        if threshold != DEFAULT_NCC_THRESHOLD:
            raise ValueError('画像非同梱×印モデルは既定閾値だけに対応します')
        path = model_path or Path(os.environ.get(MODEL_ENV, str(DEFAULT_MODEL)))
        self._student = joblib.load(path)
        if self._student.get('schema') != MODEL_SCHEMA:
            raise ValueError('画像非同梱×印モデルの形式が違います')
        from src.fast_forest import accelerate
        self._student = accelerate(self._student)     # 答えは同じまま速くする (11/14)
        self._threshold = threshold

    def match(self, bgr_patch: np.ndarray,
              context: np.ndarray | None = None) -> UiMatchResult:
        if bgr_patch.size == 0:
            return UiMatchResult(False, None, 0.0)
        name, score = self._predict_patch(bgr_patch)
        if (not name and context is not None and score >= CONTEXT_MIN_CONFIDENCE
                and _context_cross_score(context) >= CONTEXT_MIN_CORRELATION):
            return UiMatchResult(True, 'x_mark_geometry', score)
        return UiMatchResult(bool(name), name or None, score)

    def _predict_patch(self, bgr_patch: np.ndarray) -> tuple[str, float]:
        """セルの種類と確信度。同じ画素は直近の答えを使い回す (判定は画素だけで決まる)。"""
        key = bgr_patch.shape, bgr_patch.dtype.str, bgr_patch.tobytes()   # 画素の並びそのもの (取り違えがない)
        memo = self.__dict__.setdefault('_patch_memo', OrderedDict())
        if key in memo:
            memo.move_to_end(key)
            return memo[key]
        from src.fast_forest import predict_together
        labels, confidences = predict_together([self._student['labels'], self._student['confidence']],
                                               digit_features(bgr_patch).reshape(1, -1))
        memo[key] = (str(labels[0]), float(confidences[0]))
        if len(memo) > PATCH_MEMO_SIZE:
            memo.popitem(last=False)
        return memo[key]

    def is_ui(self, bgr_patch: np.ndarray) -> bool:
        return self.match(bgr_patch).is_ui
