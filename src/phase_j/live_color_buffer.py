"""所有者が残る画像を上書きしない、色補正の出力領域再利用候補。"""
from __future__ import annotations

import sys
import os
from typing import Callable

import cv2
import numpy as np

from .dshow_color import _MATRIX_BGR, correct_601_as_709

BUFFER_COUNT = 3
UNSHARED_REFS = 2  # pool内の1参照とgetrefcountの一時参照。


def configured_corrector() -> Callable[[np.ndarray], np.ndarray] | None:
    """既定OFFでは従来経路を残す。各入力源に独立のbufferを持たせる。"""
    from src.production_config import LIVE_COLOR_BUFFER_DEFAULT
    enabled = os.environ.get('PUYO_LIVE_COLOR_BUFFER', str(int(LIVE_COLOR_BUFFER_DEFAULT))) == '1'
    return ColorBuffer().correct if enabled else None


class ColorBuffer:
    """入力源の所有スレッド専用。CPythonの参照数でview/非同期保持を除外する。"""

    def __init__(self) -> None:
        self.buffers: list[np.ndarray] = []
        self.reused = 0
        self.allocated = 0

    def correct(self, image: np.ndarray) -> np.ndarray:
        """従来と同じtransform。保持中の全領域は残して独立出力を確保する。"""
        if sys.implementation.name != 'cpython':
            return correct_601_as_709(image)
        for index in range(len(self.buffers)):
            if sys.getrefcount(self.buffers[index]) != UNSHARED_REFS:
                continue
            if self.buffers[index].shape != image.shape or self.buffers[index].dtype != image.dtype:
                self.buffers[index] = np.empty_like(image)
            self.reused += 1
            return cv2.transform(image, _MATRIX_BGR, dst=self.buffers[index])
        result = correct_601_as_709(image)
        self.allocated += 1
        if len(self.buffers) < BUFFER_COUNT:
            self.buffers.append(result)
        return result
