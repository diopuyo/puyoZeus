"""読み取りを待たせず、最新の一枚だけを低頻度で検査する。"""
from __future__ import annotations

import threading
import time
from typing import Any
import numpy as np
from .live_screen_errors import record_error

PERIOD_SEC = 1.0
MAX_RESULT_AGE_SEC = 3.0
COLOR_CHANNELS = 3


def valid_image(image: Any) -> bool:
    """欠けた入力や形式変更中の画像を検査の箱へ入れない。"""
    return (isinstance(image, np.ndarray) and image.ndim == COLOR_CHANNELS
            and image.size > 0 and image.shape[2] == COLOR_CHANNELS)


class FeedbackWorker:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: Any = None
        self._closed = False
        self._result = (False, float('-inf'))
        self._generation = 0
        self.thread = threading.Thread(target=self._run, name='feedback-check', daemon=True)
        self.thread.start()

    def offer(self, image: Any, captured: float, corrected: Any = None) -> None:
        # capture と色補正の再利用バッファから所有権を切り離す。
        try:
            lazy = corrected is not None and hasattr(corrected, 'raw')
            other = corrected.raw if lazy else corrected
            if not valid_image(image) or (other is not None and not valid_image(other)):
                self.clear()
                return
            snapshot = (image.copy(), captured, None if other is None else other.copy(), lazy)
            with self._condition:
                if not self._closed:
                    self._pending = (*snapshot, self._generation)
                    self._condition.notify()
        except Exception:
            record_error('feedback_offer')
            self.clear()

    def clear(self) -> None:
        """処理中の古い判定も無効にし、次の正常な画像を待つ。"""
        with self._condition:
            self._generation += 1
            self._pending = None
            self._result = (False, float('-inf'))
            self._condition.notify()

    def result(self, now: float) -> tuple[bool, float]:
        with self._condition:
            present, checked = self._result
            return present and 0 <= now - checked <= MAX_RESULT_AGE_SEC, checked

    def close(self) -> None:
        with self._condition:
            self._closed, self._pending = True, None
            self._result = (False, float('-inf'))
            self._condition.notify()

    def _run(self) -> None:
        next_check = 0.0
        while True:
            with self._condition:
                while not self._closed:
                    delay = next_check - time.monotonic()
                    if self._pending is not None and delay <= 0:
                        break
                    self._condition.wait(max(0, delay) if self._pending is not None else None)
                if self._closed:
                    return
                image, captured, corrected, lazy, generation = self._pending
                self._pending = None
            try:
                present = self._detect(image, corrected, lazy)
            except Exception:
                record_error('feedback_detect')
                present = False
            with self._condition:
                if not self._closed and generation == self._generation:
                    self._result = (present, captured)
            next_check = time.monotonic() + PERIOD_SEC

    @staticmethod
    def _detect(image: Any, corrected: Any, lazy: bool) -> bool:
        from .distribution_input import has_feedback
        from .live_device import check_screen
        from .dshow_color import correct_601_as_709
        if check_screen('feedback', has_feedback, image):
            return True
        if corrected is None:
            return False
        if lazy:
            corrected = correct_601_as_709(corrected)
        return check_screen('feedback_corrected', has_feedback, corrected)
