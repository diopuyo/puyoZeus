"""CellStabilityTracker テスト。"""
from __future__ import annotations

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, Board, COLOR_EMPTY, COLOR_GREEN,
    COLOR_RED, HIDDEN_ROWS,
)
from src.cell_stability_tracker import CellStabilityTracker
from src.image_reader import BoardRegion

REGION = BoardRegion(x=0, y=0, width=384, height=720)


def _frame(s: int, v: int, h: int = 60) -> np.ndarray:
    hsv = np.zeros((720, 384, 3), dtype=np.uint8)
    hsv[:, :, 0] = h
    hsv[:, :, 1] = s
    hsv[:, :, 2] = v
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _full(color: int) -> Board:
    b = Board()
    for vrow in range(12):
        for col in range(BOARD_COLS):
            b.set(vrow + HIDDEN_ROWS, col, color)
    return b


def test_no_unstable_when_hsv_constant() -> None:
    """同じ HSV を window 数 push → σ=0、安定。"""
    tracker = CellStabilityTracker(window=5, threshold=10.0)
    frame = _frame(180, 200)
    board = _full(COLOR_GREEN)
    for _ in range(5):
        tracker.refine(frame, REGION, board, "1P")
    new_board, mask = tracker.refine(frame, REGION, board, "1P")
    assert not mask.any()


def test_unstable_when_hsv_oscillates() -> None:
    """振動 HSV + 履歴 majority と異なる color → 不安定検出 + 上書き。"""
    tracker = CellStabilityTracker(window=5, threshold=15.0)
    board_green = _full(COLOR_GREEN)
    board_red = _full(COLOR_RED)
    f1 = _frame(50, 100)
    f2 = _frame(200, 250)
    # 振動 HSV を 5 frame、color は GRN majority
    tracker.refine(f1, REGION, board_green, "1P")
    tracker.refine(f2, REGION, board_green, "1P")
    tracker.refine(f1, REGION, board_green, "1P")
    tracker.refine(f2, REGION, board_green, "1P")
    tracker.refine(f1, REGION, board_green, "1P")
    # 6 frame 目: 振動継続 + 突如 color RED → 不安定検出 + RED→GRN 補正
    new_board, mask = tracker.refine(f2, REGION, board_red, "1P")
    assert mask.any()
    # majority (GRN) で復元
    for vrow in range(12):
        for col in range(BOARD_COLS):
            if mask[vrow, col]:
                assert int(new_board.get(vrow + HIDDEN_ROWS, col)) == COLOR_GREEN


def test_skip_when_chain() -> None:
    tracker = CellStabilityTracker()
    frame = _frame(180, 200)
    board = _full(COLOR_GREEN)
    new_board, mask = tracker.refine(
        frame, REGION, board, "1P", is_chain=True,
    )
    assert not mask.any()


def test_reset_clears_history() -> None:
    tracker = CellStabilityTracker()
    frame = _frame(180, 200)
    board = _full(COLOR_GREEN)
    tracker.refine(frame, REGION, board, "1P")
    assert len(tracker.history) > 0
    tracker.reset()
    assert len(tracker.history) == 0
