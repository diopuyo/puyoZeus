"""動画入力境界だけを改変する再現可能な故障注入。評価値は操作しない。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import time

import cv2
import numpy as np

KINDS = ('stall', 'resolution', 'black', 'other', 'repeat')
LOW_SIZE = (1280, 720)
DEFAULT_DURATION_SEC = 20.
STALL_DURATION_SEC = 5.


@dataclass
class FaultPlan:
    kind: str
    at_sec: float
    duration_sec: float = DEFAULT_DURATION_SEC
    video: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS or not np.isfinite(self.at_sec) or self.at_sec < 0:
            raise ValueError('故障種別または開始時刻が不正です')
        if not np.isfinite(self.duration_sec) or self.duration_sec <= 0:
            raise ValueError('故障時間は正の有限値が必要です')
        if self.kind == 'other' and (not self.video or not Path(self.video).is_file()):
            raise ValueError('別動画の既存パスが必要です')


class FaultInjector:
    def __init__(self, plan: FaultPlan, emit: Callable[[dict], None] | None = None) -> None:
        self.plan, self.emit = plan, emit or (lambda event: None)
        self.started = self.finished = False
        self.frozen: np.ndarray | None = None
        self.capture: Any = None

    def event(self, phase: str, media: float) -> None:
        self.emit(dict(kind=self.plan.kind, phase=phase, t_sec=media, at=time.perf_counter()))

    def before(self, media: float, sleep: Callable, on_hold: Callable | None) -> None:
        if self.plan.kind != 'stall' or self.started or media < self.plan.at_sec:
            return
        self.started = True
        self.event('start', media)
        if on_hold:
            on_hold('verifying')
        sleep(self.plan.duration_sec)
        self.finished = True
        self.event('end', media+self.plan.duration_sec)

    def transform(self, image: np.ndarray, media: float) -> np.ndarray:
        if self.plan.kind == 'stall' or media < self.plan.at_sec:
            return image
        if media >= self.plan.at_sec+self.plan.duration_sec:
            if self.started and not self.finished:
                self.event('end', media)
                self.finished = True
                self.close()
            return image
        if not self.started:
            self.started = True
            self.event('start', media)
        if self.plan.kind == 'resolution':
            return cv2.resize(image, LOW_SIZE)
        if self.plan.kind == 'black':
            return np.zeros_like(image)
        if self.plan.kind == 'repeat':
            if self.frozen is None:
                self.frozen = image.copy()
            return self.frozen.copy()
        if self.capture is None:
            self.capture = cv2.VideoCapture(self.plan.video)
        self.capture.set(cv2.CAP_PROP_POS_MSEC, (media-self.plan.at_sec)*1000)
        ok, alternate = self.capture.read()
        if not ok:
            raise EOFError('故障用別動画が不足しています')
        return alternate

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
