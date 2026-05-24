"""CellHsvAnomalyDetector のテスト (Z-3J')。"""
from __future__ import annotations

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, Board, COLOR_EMPTY, COLOR_GREEN,
    COLOR_RED, HIDDEN_ROWS,
)
from src.cell_hsv_anomaly_detector import CellHsvAnomalyDetector
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


def test_no_anomaly_first_frame() -> None:
    detector = CellHsvAnomalyDetector(window=3)
    frame = _frame(180, 200)
    board = _full(COLOR_GREEN)
    new_board, mask = detector.refine(frame, REGION, board, "1P")
    assert not mask.any()


def test_no_anomaly_when_hsv_stable() -> None:
    detector = CellHsvAnomalyDetector(window=3, threshold=20.0)
    frame = _frame(180, 200)
    board = _full(COLOR_GREEN)
    for _ in range(3):
        detector.refine(frame, REGION, board, "1P")
    new_board, mask = detector.refine(frame, REGION, board, "1P")
    assert not mask.any()


def test_anomaly_when_hsv_changes_drastically() -> None:
    """S/V が大きく変わったら anomaly 検出。"""
    detector = CellHsvAnomalyDetector(window=3, threshold=20.0)
    stable = _frame(180, 200)
    board = _full(COLOR_GREEN)
    for _ in range(3):
        detector.refine(stable, REGION, board, "1P")
    # S を 100 に減らす (距離 80) → anomaly
    different = _frame(80, 200)
    new_board, mask = detector.refine(
        different, REGION, _full(COLOR_RED), "1P",
    )
    assert mask.any()


def test_no_anomaly_when_hsv_small_variation() -> None:
    """puyo 自然変動 (距離 < threshold) は anomaly 検出しない。"""
    detector = CellHsvAnomalyDetector(window=3, threshold=35.0)
    stable = _frame(180, 200)
    board = _full(COLOR_GREEN)
    for _ in range(3):
        detector.refine(stable, REGION, board, "1P")
    # 自然変動: S 180→195 (距離 15) → anomaly なし
    natural = _frame(195, 200)
    new_board, mask = detector.refine(natural, REGION, board, "1P")
    assert not mask.any()


def test_skip_when_chain() -> None:
    detector = CellHsvAnomalyDetector()
    frame = _frame(180, 200)
    board = _full(COLOR_GREEN)
    new_board, mask = detector.refine(
        frame, REGION, board, "1P", is_chain=True,
    )
    assert not mask.any()


def test_anomaly_replaces_with_stable_color() -> None:
    detector = CellHsvAnomalyDetector(window=3, threshold=20.0)
    stable = _frame(180, 200)
    green_board = _full(COLOR_GREEN)
    for _ in range(3):
        detector.refine(stable, REGION, green_board, "1P")
    different = _frame(80, 200)
    red_board = _full(COLOR_RED)
    new_board, mask = detector.refine(
        different, REGION, red_board, "1P",
    )
    for vrow in range(12):
        for col in range(BOARD_COLS):
            if mask[vrow, col]:
                assert int(new_board.get(vrow + HIDDEN_ROWS, col)) == COLOR_GREEN


def test_reset_clears() -> None:
    detector = CellHsvAnomalyDetector()
    frame = _frame(180, 200)
    board = _full(COLOR_GREEN)
    detector.refine(frame, REGION, board, "1P")
    assert len(detector.history) > 0
    detector.reset()
    assert len(detector.history) == 0
