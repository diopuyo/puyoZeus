"""B4 StableNextDetector のテスト。"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np

from src.next_detector import (
    NextDetectionBothResult,
    NextDetectionResult,
)
from src.stable_next_detector import StableNextDetector


def _make_result(
    p1_top, p1_bot, p2_top, p2_bot,
    p1_dt=0, p1_db=0, p2_dt=0, p2_db=0,
) -> NextDetectionBothResult:
    return NextDetectionBothResult(
        p1=NextDetectionResult(
            next_top=p1_top, next_bot=p1_bot,
            dnext_top=p1_dt, dnext_bot=p1_db,
        ),
        p2=NextDetectionResult(
            next_top=p2_top, next_bot=p2_bot,
            dnext_top=p2_dt, dnext_bot=p2_db,
        ),
    )


def test_stable_after_consecutive_match() -> None:
    """3 連続同じ next_pair → 採用。"""
    base = MagicMock()
    base.detect_both.return_value = _make_result(1, 2, 1, 2)
    det = StableNextDetector(base, stability_window=3)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    r1 = det.detect_both(frame)
    assert r1.p1_next is None  # window 未満
    r2 = det.detect_both(frame)
    assert r2.p1_next is None
    r3 = det.detect_both(frame)
    assert r3.p1_next == (1, 2)  # 3 連続で採用
    assert r3.p2_next == (1, 2)


def test_unstable_keeps_previous() -> None:
    """安定後に不一致が来ても前回値保持。"""
    base = MagicMock()
    det = StableNextDetector(base, stability_window=3)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # 最初 3 連続 (1, 2)
    base.detect_both.return_value = _make_result(1, 2, 1, 2)
    det.detect_both(frame)
    det.detect_both(frame)
    r3 = det.detect_both(frame)
    assert r3.p1_next == (1, 2)
    # 次に不一致 (3, 4)
    base.detect_both.return_value = _make_result(3, 4, 3, 4)
    r4 = det.detect_both(frame)
    # まだ window 内に古い値が残るので、不一致 → stable 値は更新されず
    assert r4.p1_next == (1, 2)


def test_window_2_quicker() -> None:
    """window=2 なら 2 連続で採用。"""
    base = MagicMock()
    base.detect_both.return_value = _make_result(5, 1, 5, 1)
    det = StableNextDetector(base, stability_window=2)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det.detect_both(frame)
    r2 = det.detect_both(frame)
    assert r2.p1_next == (5, 1)


def test_reset_clears_history() -> None:
    base = MagicMock()
    base.detect_both.return_value = _make_result(1, 2, 1, 2)
    det = StableNextDetector(base, stability_window=2)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det.detect_both(frame)
    det.detect_both(frame)
    det.reset()
    # reset 後 1 回目は None
    base.detect_both.return_value = _make_result(3, 4, 3, 4)
    r = det.detect_both(frame)
    assert r.p1_next is None


def test_dnext_stability() -> None:
    """dnext も独立に安定化。"""
    base = MagicMock()
    base.detect_both.return_value = _make_result(
        1, 2, 1, 2, p1_dt=3, p1_db=4, p2_dt=3, p2_db=4,
    )
    det = StableNextDetector(base, stability_window=2)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det.detect_both(frame)
    r = det.detect_both(frame)
    assert r.p1_dnext == (3, 4)
    assert r.p2_dnext == (3, 4)
