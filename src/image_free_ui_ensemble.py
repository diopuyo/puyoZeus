"""画像を保持しないUI分類器。分類対象と座標回帰を分離する。"""
from __future__ import annotations
from src.phase_j.live_color_roi import full_image

import json
from typing import Any

import cv2
import numpy as np

from src.image_free_digit import digit_features

SPATIAL_SIZE = (48, 40)
COLOR_SIZE = (16, 16)
COLOR_CHANNEL_MAX = 255.
DECISION_MARGIN = 0.01


def task_features(key: str, image: np.ndarray, color: bool = False) -> np.ndarray:
    """死亡は対象側だけを見て、他方の盤面や得点への依存を防ぐ。"""
    from src.match_end_detector import SEARCH_P1, SEARCH_P2
    regions = (SEARCH_P1, SEARCH_P2)
    if key.startswith('death_'):
        regions = (regions[int(key[-1])],)
    if key in ('chain', 'ojama'):
        patches = [image]
    else:
        patches = [image[y:y+h, x:x+w] for x, y, w, h in regions]
    values = [spatial_features(patch) for patch in patches]
    if color:
        # 予告の旧教師はBGR照合。濃淡だけでは異なる色のアイコンを区別できない。
        values.append(cv2.resize(full_image(image), COLOR_SIZE, interpolation=cv2.INTER_AREA).ravel().astype(np.float32)/COLOR_CHANNEL_MAX)
    return np.concatenate(values)


def spatial_features(image: np.ndarray) -> np.ndarray:
    """広い領域を桁サイズまで潰さず、濃淡と勾配を観測する。"""
    gray = cv2.cvtColor(full_image(image), cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, SPATIAL_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    small = (small - small.mean()) / max(float(small.std()), 1.)
    return np.concatenate((small.ravel(), digit_features(gray)))


class UIEnsemble:
    """各クラスの投票で判断し、座標をクラス数へ混入させない。"""

    def __init__(self, key: str, classifier: Any, coordinates: Any = None, color: bool = False) -> None:
        self.key, self.classifier, self.coordinates = key, classifier, coordinates
        self.color = color

    def predict_image(self, image: np.ndarray) -> Any:
        from src.match_end_detector import DEFAULT_NCC_THRESHOLD
        from src.chain_count_ocr import CHAIN_NCC_MIN_CONFIDENCE
        data = task_features(self.key, image, getattr(self, 'color', False)).reshape(1, -1)
        label = json.loads(str(self.classifier.predict(data)[0]))
        if self.key == 'ojama':
            return label
        if self.key.startswith('death_'):
            return DEFAULT_NCC_THRESHOLD + (DECISION_MARGIN if label else -DECISION_MARGIN)
        if self.key == 'match_end':
            return dict(detected=label is not None, template_name=label,
                        score=DEFAULT_NCC_THRESHOLD + (DECISION_MARGIN if label else -DECISION_MARGIN))
        location = None
        if label is not None:
            location = tuple(map(int, np.rint(self.coordinates.predict(data)[0])))
        return dict(chain_count=label, location=location,
                    confidence=CHAIN_NCC_MIN_CONFIDENCE + (DECISION_MARGIN if label else -DECISION_MARGIN))
