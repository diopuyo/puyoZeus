"""CellAnomalyDetector のテスト (Z-3J)。"""
from __future__ import annotations

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, Board, COLOR_EMPTY, COLOR_GREEN,
    COLOR_RED, HIDDEN_ROWS,
)
from src.cell_anomaly_detector import CellAnomalyDetector
from src.image_reader import BoardRegion

REGION = BoardRegion(x=0, y=0, width=384, height=720)


def _frame(color_bgr: tuple[int, int, int]) -> np.ndarray:
    f = np.full((720, 384, 3), color_bgr, dtype=np.uint8)
    return f


def _make_em_board() -> Board:
    return Board()


def _full_color_board(color: int) -> Board:
    board = Board()
    for vrow in range(12):
        for col in range(BOARD_COLS):
            board.set(vrow + HIDDEN_ROWS, col, color)
    return board


def test_no_anomaly_on_first_frame() -> None:
    """history が空なら anomaly 検出されない (= history 構築フェーズ)。"""
    detector = CellAnomalyDetector(window=3)
    frame = _frame((100, 100, 100))
    board = _full_color_board(COLOR_GREEN)
    new_board, mask = detector.refine(frame, REGION, board, "1P")
    assert not mask.any()


def test_no_anomaly_when_hash_stable() -> None:
    """同じ frame を window 数 push → hash 安定 → 次 frame も anomaly なし。"""
    detector = CellAnomalyDetector(window=3)
    frame = _frame((100, 100, 100))
    board = _full_color_board(COLOR_GREEN)
    # window 数だけ history 蓄積
    for _ in range(3):
        detector.refine(frame, REGION, board, "1P")
    # 同じ frame で再度 → anomaly なし
    new_board, mask = detector.refine(frame, REGION, board, "1P")
    assert not mask.any()


def test_anomaly_when_patch_changes_drastically() -> None:
    """突然 patch が劇的に変化したら anomaly 検出。

    cell ごとに patch の grayscale 分布が変わるパターン (gradient + noise)
    を作って dHash が大きく変動するようにする。
    """
    detector = CellAnomalyDetector(window=3, threshold=5)
    rng = np.random.default_rng(42)
    # window 分は同じ pattern (random だが seed 固定で同一 frame)
    stable_rng = np.random.default_rng(0)
    stable = stable_rng.integers(
        0, 256, size=(720, 384, 3), dtype=np.uint8,
    )
    board = _full_color_board(COLOR_GREEN)
    for _ in range(3):
        detector.refine(stable, REGION, board, "1P")
    # 突然全く違う pattern (別 seed の random)
    different = rng.integers(0, 256, size=(720, 384, 3), dtype=np.uint8)
    new_board, mask = detector.refine(
        different, REGION, _full_color_board(COLOR_RED), "1P",
    )
    # 一部 cell で anomaly 発火
    assert mask.any()


def test_skip_when_chain() -> None:
    """is_chain=True なら anomaly チェック skip。"""
    detector = CellAnomalyDetector(window=3)
    frame = _frame((100, 100, 100))
    board = _full_color_board(COLOR_GREEN)
    new_board, mask = detector.refine(
        frame, REGION, board, "1P", is_chain=True,
    )
    assert not mask.any()


def test_anomaly_cell_color_replaced() -> None:
    """anomaly cell の色が history 最頻色で置き換えられる。"""
    detector = CellAnomalyDetector(window=3, threshold=5)
    stable_rng = np.random.default_rng(0)
    stable = stable_rng.integers(
        0, 256, size=(720, 384, 3), dtype=np.uint8,
    )
    green_board = _full_color_board(COLOR_GREEN)
    # 3 回 GRN で蓄積
    for _ in range(3):
        detector.refine(stable, REGION, green_board, "1P")
    # 突然全く違う patch + RED 認識
    rng = np.random.default_rng(42)
    different = rng.integers(0, 256, size=(720, 384, 3), dtype=np.uint8)
    red_board = _full_color_board(COLOR_RED)
    new_board, mask = detector.refine(
        different, REGION, red_board, "1P",
    )
    # anomaly した cell は GRN (history 最頻色) に戻る
    for vrow in range(12):
        for col in range(BOARD_COLS):
            if mask[vrow, col]:
                assert int(new_board.get(vrow + HIDDEN_ROWS, col)) == COLOR_GREEN


def test_reset_clears_history() -> None:
    detector = CellAnomalyDetector(window=3)
    frame = _frame((100, 100, 100))
    board = _full_color_board(COLOR_GREEN)
    detector.refine(frame, REGION, board, "1P")
    assert len(detector.history) > 0
    detector.reset()
    assert len(detector.history) == 0
