"""C1: 使用領域だけを補正し、全画面消費者には遅延して全画面を渡す。"""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np
from src.live_speed_c import enabled
from .dshow_color import _MATRIX_BGR, correct_601_as_709

STATS = dict(frames=0, regions=0, full=0)
_INVERSE_BGR = np.linalg.inv(_MATRIX_BGR.astype(np.float64)).astype(np.float32)


class RegionCorrectedImage:
    """未補正の入力を所有する。切出し結果は通常 ndarray、全画面要求は一回だけ。"""

    def __init__(self, image: np.ndarray) -> None:
        self.raw = image.copy()
        self.complete: np.ndarray | None = None
        self.native: np.ndarray | None = None
        self.shape, self.dtype, self.ndim, self.size = image.shape, image.dtype, image.ndim, image.size
        STATS['frames'] += 1

    def full(self) -> np.ndarray:
        """全画面を読む OpenCV、NumPy、保存処理で共通使用する。"""
        if self.complete is None:
            self.complete = correct_601_as_709(self.raw)
            STATS['full'] += 1
        return self.complete

    def __getitem__(self, key: Any) -> np.ndarray:
        """通常の矩形・間引き参照はその画素だけ補正する。特殊添字は全画面へ。"""
        if self.complete is not None:
            return self.complete[key]
        if (isinstance(key, tuple) and len(key) == 2
                and all(isinstance(k, slice) for k in key)):
            crop = self.raw[key]
            if crop.size == 0:
                return crop.copy()
            STATS['regions'] += 1
            return correct_601_as_709(crop)
        return self.full()[key]

    def native_regions(self, rects: Any) -> np.ndarray:
        """Rust first_pass は矩形表の内側だけ読む。外側を未補正のまま保持する。"""
        if self.complete is not None:
            return self.complete
        if not len(rects):
            return self.full()
        boxes = np.asarray(rects)
        x1, y1 = boxes[:, :2].min(axis=0)
        x2, y2 = boxes[:, 2:].max(axis=0)
        if not (0 <= x1 < x2 <= self.shape[1] and 0 <= y1 < y2 <= self.shape[0]):
            return self.full()
        if self.native is None:
            self.native = self.raw.copy()
        self.native[y1:y2, x1:x2] = self[y1:y2, x1:x2]
        return self.native

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """NumPy の全画面変換は省略しない。"""
        result = np.asarray(self.full(), dtype=dtype)
        return result.copy() if copy else result

    def copy(self, order: str = 'C') -> np.ndarray:
        """背景学習・非同期検証・画像保存は独立した全画面を保持する。"""
        return self.full().copy(order=order)


def full_image(image: Any) -> Any:
    """通常画像を無変更で返す。C1 の遅延画像だけ全画面を補正する。"""
    return image.full() if isinstance(image, RegionCorrectedImage) else image


def correct_input(image: np.ndarray) -> Any:
    """両腕とも補正 ON。候補だけ使用時まで領域補正を遅らせる。"""
    return RegionCorrectedImage(image) if enabled('live_color_roi') else correct_601_as_709(image)


def simulate_dshow(image: np.ndarray) -> np.ndarray:
    """検証専用の709→601逆変換。丸め・飽和は戻らないので品質評価には使わない。"""
    return cv2.transform(image, _INVERSE_BGR)


def native_image(image: Any, rects: Any) -> Any:
    """矩形表を使う Rust 入口専用。他の全画面処理へは渡さない。"""
    return image.native_regions(rects) if isinstance(image, RegionCorrectedImage) else image
