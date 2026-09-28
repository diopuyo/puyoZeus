"""故障注入の有無によらず、入力形式と映像の進行を監視する。"""
from __future__ import annotations

import hashlib
import numpy as np

FINGERPRINT_STRIDE = 8


class FrameContinuity:
    """入力セッションの形式を固定し、異常復帰後も一定時間再確認する。"""

    def __init__(self, period_sec: float) -> None:
        self.period_sec = period_sec
        self.reference_size: tuple[int, int] | None = None
        self.digest: bytes | None = None
        self.changed_at = 0.
        self.recovering = False
        self.verify_until = float('-inf')

    def ready(self, image: np.ndarray, size: tuple[int, int], stamp: float) -> bool:
        if self.reference_size is None:
            self.reference_size = size
        digest = hashlib.blake2b(image[::FINGERPRINT_STRIDE, ::FINGERPRINT_STRIDE].tobytes()).digest()
        if digest != self.digest:
            self.digest, self.changed_at = digest, stamp
        wrong_size = size != self.reference_size
        repeated = stamp-self.changed_at >= self.period_sec
        if wrong_size or repeated:
            self.recovering = True
        elif self.recovering:
            self.recovering = False
            self.verify_until = stamp+self.period_sec
        return not (wrong_size or repeated or stamp < self.verify_until)
