"""窒息位置の×印を画像非同梱の小型分類器で判定する未採用経路。"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np

from src.image_free_digit import digit_features
from src.ui_mask import UiMaskMatcher, UiMatchResult, DEFAULT_NCC_THRESHOLD

MODEL_SCHEMA = 1
MODEL_ENV = 'PUYO_IMAGE_FREE_UI_MASK_MODEL'
DEFAULT_MODEL = Path('models/image_free/ui_mask.joblib')


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
        self._threshold = threshold

    def match(self, bgr_patch: np.ndarray) -> UiMatchResult:
        if bgr_patch.size == 0:
            return UiMatchResult(False, None, 0.0)
        features = digit_features(bgr_patch).reshape(1, -1)
        name = str(self._student['labels'].predict(features)[0])
        score = float(self._student['confidence'].predict(features)[0])
        return UiMatchResult(bool(name), name or None, score)

    def is_ui(self, bgr_patch: np.ndarray) -> bool:
        return self.match(bgr_patch).is_ui
