"""B4: NextDetector を wrap し、連続 N フレーム同色一致時のみ next_pair を採用。

NextDetector 単発判定では P1/P2 一致率 25% (ほぼランダム)。
時系列で「連続 N フレーム同色一致」のときのみ stable と判定し、
不一致時は前回 stable 値を保持する。

これにより V2.1 NextLinkedColorRefiner / W3.0 hidden_row_inferrer に
信頼できる next_pair のみを供給できる。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.next_detector import (
    NextDetectionBothResult,
    NextDetectionResult,
    NextDetector,
)


DEFAULT_STABILITY_WINDOW: int = 3


@dataclass(frozen=True)
class StableDetectionBoth:
    """安定検出結果。stable_p1/p2 は None 可 (まだ安定してない場合)。"""
    p1_next: tuple[int, int] | None
    p2_next: tuple[int, int] | None
    p1_dnext: tuple[int, int] | None
    p2_dnext: tuple[int, int] | None
    raw: NextDetectionBothResult  # 元の raw 検出結果も保持


class StableNextDetector:
    """NextDetector を wrap して連続 N フレーム同色のみ採用。"""

    def __init__(
        self,
        base: NextDetector,
        stability_window: int = DEFAULT_STABILITY_WINDOW,
    ) -> None:
        self._base = base
        self._window = int(stability_window)
        self._history_p1_next: deque[tuple[int, int]] = deque(
            maxlen=self._window,
        )
        self._history_p2_next: deque[tuple[int, int]] = deque(
            maxlen=self._window,
        )
        self._history_p1_dnext: deque[tuple[int, int]] = deque(
            maxlen=self._window,
        )
        self._history_p2_dnext: deque[tuple[int, int]] = deque(
            maxlen=self._window,
        )
        self._stable_p1_next: tuple[int, int] | None = None
        self._stable_p2_next: tuple[int, int] | None = None
        self._stable_p1_dnext: tuple[int, int] | None = None
        self._stable_p2_dnext: tuple[int, int] | None = None

    def reset(self) -> None:
        self._history_p1_next.clear()
        self._history_p2_next.clear()
        self._history_p1_dnext.clear()
        self._history_p2_dnext.clear()
        self._stable_p1_next = None
        self._stable_p2_next = None
        self._stable_p1_dnext = None
        self._stable_p2_dnext = None

    def detect_both(self, frame: np.ndarray) -> StableDetectionBoth:
        """フレーム検出 + 安定値更新。"""
        raw = self._base.detect_both(frame)
        # 履歴追加
        self._history_p1_next.append(raw.p1.next_pair)
        self._history_p2_next.append(raw.p2.next_pair)
        self._history_p1_dnext.append(raw.p1.dnext_pair)
        self._history_p2_dnext.append(raw.p2.dnext_pair)
        # 安定値更新 (window 全部同じなら採用)
        if len(self._history_p1_next) == self._window:
            if self._all_same(self._history_p1_next):
                self._stable_p1_next = self._history_p1_next[0]
            if self._all_same(self._history_p2_next):
                self._stable_p2_next = self._history_p2_next[0]
            if self._all_same(self._history_p1_dnext):
                self._stable_p1_dnext = self._history_p1_dnext[0]
            if self._all_same(self._history_p2_dnext):
                self._stable_p2_dnext = self._history_p2_dnext[0]
        return StableDetectionBoth(
            p1_next=self._stable_p1_next,
            p2_next=self._stable_p2_next,
            p1_dnext=self._stable_p1_dnext,
            p2_dnext=self._stable_p2_dnext,
            raw=raw,
        )

    @staticmethod
    def _all_same(deq: "deque[tuple[int, int]]") -> bool:
        if not deq:
            return False
        first = deq[0]
        return all(d == first for d in deq)


__all__ = [
    "DEFAULT_STABILITY_WINDOW",
    "StableDetectionBoth",
    "StableNextDetector",
]
