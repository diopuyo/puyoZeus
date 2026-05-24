"""DriftDetector テスト (Phase B-6)."""

from __future__ import annotations

import pytest

from src.board import COLOR_BLUE, COLOR_RED, COLOR_UNKNOWN, Board
from src.drift_detector import (
    DEFAULT_DRIFT_CELL_THRESHOLD,
    DEFAULT_DRIFT_FRAME_THRESHOLD,
    DriftDetector,
    DriftResult,
)


# ============================
# helper
# ============================


def _empty_board() -> Board:
    return Board()


def _board_with_n_red_cells(n: int) -> Board:
    """row=12, col=0..n-1 に赤を配置."""
    b = Board()
    for c in range(min(n, 6)):
        b.set(12, c, COLOR_RED)
    return b


# ============================
# 基本動作
# ============================


def test_match_returns_no_drift() -> None:
    det = DriftDetector(cell_threshold=2, frame_threshold=3)
    a = _board_with_n_red_cells(3)
    b = a.copy()
    res = det.update(a, b)
    assert isinstance(res, DriftResult)
    assert res.mismatch_count == 0
    assert res.is_drift is False
    assert res.needs_resync is False


def test_small_mismatch_below_threshold() -> None:
    det = DriftDetector(cell_threshold=3, frame_threshold=3)
    a = _board_with_n_red_cells(0)
    b = _board_with_n_red_cells(2)  # 2 cell 違い
    res = det.update(a, b)
    assert res.mismatch_count == 2
    assert res.is_drift is False


def test_drift_above_cell_threshold() -> None:
    det = DriftDetector(cell_threshold=3, frame_threshold=3)
    a = _board_with_n_red_cells(0)
    b = _board_with_n_red_cells(5)
    res = det.update(a, b)
    assert res.mismatch_count == 5
    assert res.is_drift is True
    assert res.consecutive_count == 1
    assert res.needs_resync is False


def test_consecutive_drift_triggers_resync() -> None:
    det = DriftDetector(cell_threshold=2, frame_threshold=3)
    a = _empty_board()
    b = _board_with_n_red_cells(4)
    r1 = det.update(a, b)
    r2 = det.update(a, b)
    r3 = det.update(a, b)
    assert r1.needs_resync is False
    assert r2.needs_resync is False
    assert r3.needs_resync is True
    assert r3.consecutive_count == 3


def test_drift_clears_on_match() -> None:
    det = DriftDetector(cell_threshold=2, frame_threshold=3)
    a = _empty_board()
    drifted = _board_with_n_red_cells(4)
    det.update(a, drifted)
    det.update(a, drifted)
    assert det.consecutive_drift_count == 2
    # 次 frame で一致
    res = det.update(a, a)
    assert res.is_drift is False
    assert res.consecutive_count == 0


def test_unknown_cells_excluded_from_mismatch() -> None:
    det = DriftDetector(cell_threshold=1, frame_threshold=3)
    a = _empty_board()
    b = Board()
    # b にだけ UNKNOWN を入れる: mismatch にカウントしない
    b.set(12, 0, COLOR_UNKNOWN)
    res = det.update(a, b)
    assert res.mismatch_count == 0
    assert res.is_drift is False


def test_none_input_returns_no_drift() -> None:
    det = DriftDetector(cell_threshold=1, frame_threshold=2)
    res1 = det.update(None, _empty_board())
    res2 = det.update(_empty_board(), None)
    res3 = det.update(None, None)
    for r in (res1, res2, res3):
        assert r.is_drift is False
        assert r.needs_resync is False


def test_reset_clears_consecutive_count() -> None:
    det = DriftDetector(cell_threshold=2, frame_threshold=3)
    a = _empty_board()
    b = _board_with_n_red_cells(4)
    det.update(a, b)
    det.update(a, b)
    assert det.consecutive_drift_count == 2
    det.reset()
    assert det.consecutive_drift_count == 0


# ============================
# default 定数
# ============================


def test_defaults() -> None:
    det = DriftDetector()
    assert det.cell_threshold == DEFAULT_DRIFT_CELL_THRESHOLD
    assert det.frame_threshold == DEFAULT_DRIFT_FRAME_THRESHOLD


# ============================
# バリデーション
# ============================


def test_invalid_cell_threshold_raises() -> None:
    with pytest.raises(ValueError):
        DriftDetector(cell_threshold=0)


def test_invalid_frame_threshold_raises() -> None:
    with pytest.raises(ValueError):
        DriftDetector(frame_threshold=0)


# ============================
# 異色置換 (red↔blue) も乖離としてカウント
# ============================


def test_color_swap_counts_as_mismatch() -> None:
    det = DriftDetector(cell_threshold=2, frame_threshold=3)
    a = Board()
    b = Board()
    for c in range(3):
        a.set(12, c, COLOR_RED)
        b.set(12, c, COLOR_BLUE)
    res = det.update(a, b)
    assert res.mismatch_count == 3
    assert res.is_drift is True
