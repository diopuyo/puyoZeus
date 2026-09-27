"""B2専用の入力境界。認識前だけlatest-winsで間引く。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterator
import math
import time

import cv2
import numpy as np

NATIVE_SIZE = (1920, 1080)
RECOGNITION_HZ = 30
FPS_TOLERANCE = 1e-6


@dataclass(frozen=True)
class CapturedFrame:
    """capture時刻は実時間モードでは元映像が利用可能になった壁時計。"""

    index: int
    media_sec: float
    captured_at: float
    acquired_at: float
    image: np.ndarray
    dropped_before: int = 0


class FrameSource(ABC):
    """キャプチャ方式に依存しない単一のフレーム供給契約。"""

    dropped: int = 0

    @abstractmethod
    def __iter__(self) -> Iterator[CapturedFrame]:
        """利用可能になった画像を、時刻の昇順で供給する。"""
        raise NotImplementedError


class VideoFileSource(FrameSource):
    """動画を30Hzへ正規化し、遅延時は古い認識対象をデコードだけして捨てる。"""

    def __init__(self, capture: Any, fps: float, start: int, end: int,
                 stride: int, realtime: bool = False,
                 clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if fps <= 0 or stride < 1 or fps / stride > RECOGNITION_HZ + FPS_TOLERANCE:
            raise ValueError('入力の正規化周波数が不正です')
        self.capture, self.fps = capture, fps
        self.start, self.end, self.stride = start, end, stride
        self.realtime, self.clock, self.sleep = realtime, clock, sleep
        self.dropped = 0
        self.normalization_skipped = 0

    def latest_index(self, next_index: int, origin: float) -> int:
        """まだ到来していない画像へ進まない。終端の次のslotも許し全残りを捨てる。"""
        elapsed = max(0.0, self.clock() - origin)
        due = self.start + math.floor(elapsed * self.fps / self.stride) * self.stride
        limit = self.start + math.ceil((self.end - self.start) / self.stride) * self.stride
        return max(next_index, min(due, limit))

    def __iter__(self) -> Iterator[CapturedFrame]:
        origin = self.clock()
        index, next_index = self.start, self.start
        while next_index < self.end:
            target = self.latest_index(next_index, origin) if self.realtime else next_index
            dropped = (target - next_index) // self.stride
            self.dropped += dropped
            while index < min(target, self.end):
                if not self.capture.grab():
                    raise EOFError(f'予定区間の途中で入力終了: {index}')
                self.normalization_skipped += int((index - self.start) % self.stride != 0)
                index += 1
            if target >= self.end:
                break
            scheduled = origin + (target - self.start) / self.fps
            if self.realtime:
                self.sleep(max(0.0, scheduled - self.clock()))
            captured = scheduled if self.realtime else self.clock()
            ok, image = self.capture.read()
            if not ok or image is None:
                raise EOFError(f'予定区間の途中で入力終了: {target}')
            index, next_index = target + 1, target + self.stride
            if image.shape[:2] != NATIVE_SIZE[::-1]:
                image = cv2.resize(image, NATIVE_SIZE, interpolation=cv2.INTER_AREA)
            yield CapturedFrame(target, target / self.fps, captured,
                                self.clock(), image, dropped)
