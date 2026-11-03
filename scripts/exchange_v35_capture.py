"""採取専用の連続復号。指定コマと分類器は変えず、近接seekだけを省く。"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any
import cv2
from scripts.exchange_counts_v33_collect import ImageSources

CACHE_FRAMES = 8


class SequentialCapture:
    def __init__(self, cap: Any) -> None:
        self.cap = cap
        self.requested = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.next_index = self.requested
        self.cache: OrderedDict = OrderedDict()
        self.max_skip = int(cap.get(cv2.CAP_PROP_FPS))

    def get(self, prop: int) -> float:
        return self.cap.get(prop)

    def set(self, prop: int, value: float) -> bool:
        if prop != cv2.CAP_PROP_POS_FRAMES:
            return self.cap.set(prop, value)
        self.requested = int(value)
        return True

    def read(self) -> tuple[bool, Any]:
        index = self.requested
        if index in self.cache:
            return True, self.cache[index]
        if not self.next_index <= index <= self.next_index+self.max_skip:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.next_index = index
        while self.next_index <= index:
            ok, frame = self.cap.read()
            if not ok:
                return False, frame
            self.cache[self.next_index] = frame
            self.next_index += 1
            if len(self.cache) > CACHE_FRAMES:
                self.cache.popitem(last=False)
        return True, self.cache[index]

    def release(self) -> None:
        self.cap.release()


def image_sources(path: Any) -> ImageSources:
    result = ImageSources(path)
    result.cap = SequentialCapture(result.cap)
    return result
