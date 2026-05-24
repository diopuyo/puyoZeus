"""ConnectivityOutlierRefiner テスト。"""
from __future__ import annotations

import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, Board, COLOR_BLUE, COLOR_EMPTY,
    COLOR_GREEN, COLOR_OJAMA, COLOR_RED, COLOR_YELLOW, HIDDEN_ROWS,
)
from src.connectivity_outlier_refiner import ConnectivityOutlierRefiner


def _set(b: Board, vrow: int, col: int, color: int) -> None:
    b.set(vrow + HIDDEN_ROWS, col, color)


def test_isolated_cell_refined_to_neighbors_color() -> None:
    """周囲 4 つが同色 → 自身も補正。"""
    b = Board()
    _set(b, 5, 1, COLOR_GREEN)  # 上
    _set(b, 7, 1, COLOR_GREEN)  # 下
    _set(b, 6, 0, COLOR_GREEN)  # 左
    _set(b, 6, 2, COLOR_GREEN)  # 右
    _set(b, 6, 1, COLOR_RED)    # 中央 (異色)
    refiner = ConnectivityOutlierRefiner(min_neighbor_same=3)
    new_board, mask = refiner.refine(b)
    assert int(new_board.get(6 + HIDDEN_ROWS, 1)) == COLOR_GREEN
    assert mask[6 + HIDDEN_ROWS, 1]


def test_no_refine_when_few_neighbors() -> None:
    """周囲 puyo が 2 つだけ → 補正しない (min_neighbor_same=3)。"""
    b = Board()
    _set(b, 5, 1, COLOR_GREEN)
    _set(b, 7, 1, COLOR_GREEN)
    _set(b, 6, 1, COLOR_RED)
    refiner = ConnectivityOutlierRefiner(min_neighbor_same=3)
    new_board, mask = refiner.refine(b)
    assert int(new_board.get(6 + HIDDEN_ROWS, 1)) == COLOR_RED
    assert not mask[6 + HIDDEN_ROWS, 1]


def test_no_refine_when_color_matches() -> None:
    """周囲と同色 → 補正不要。"""
    b = Board()
    _set(b, 5, 1, COLOR_GREEN)
    _set(b, 7, 1, COLOR_GREEN)
    _set(b, 6, 0, COLOR_GREEN)
    _set(b, 6, 1, COLOR_GREEN)
    refiner = ConnectivityOutlierRefiner()
    new_board, mask = refiner.refine(b)
    assert not mask.any()


def test_skip_during_chain() -> None:
    """連鎖中は補正対象外。"""
    b = Board()
    _set(b, 5, 1, COLOR_GREEN)
    _set(b, 7, 1, COLOR_GREEN)
    _set(b, 6, 0, COLOR_GREEN)
    _set(b, 6, 2, COLOR_GREEN)
    _set(b, 6, 1, COLOR_RED)
    refiner = ConnectivityOutlierRefiner()
    new_board, mask = refiner.refine(b, is_chain=True)
    assert int(new_board.get(6 + HIDDEN_ROWS, 1)) == COLOR_RED
    assert not mask.any()


def test_em_cells_not_modified() -> None:
    """EM cell は対象外 (検出漏れではなく真の空とみなす)。"""
    b = Board()
    _set(b, 5, 1, COLOR_GREEN)
    _set(b, 7, 1, COLOR_GREEN)
    _set(b, 6, 0, COLOR_GREEN)
    # 中央 EM
    refiner = ConnectivityOutlierRefiner()
    new_board, mask = refiner.refine(b)
    assert int(new_board.get(6 + HIDDEN_ROWS, 1)) == COLOR_EMPTY


def test_ojama_not_modified() -> None:
    """OJM 孤立は正常 (補正しない)。"""
    b = Board()
    _set(b, 5, 1, COLOR_GREEN)
    _set(b, 7, 1, COLOR_GREEN)
    _set(b, 6, 0, COLOR_GREEN)
    _set(b, 6, 2, COLOR_GREEN)
    _set(b, 6, 1, COLOR_OJAMA)
    refiner = ConnectivityOutlierRefiner()
    new_board, mask = refiner.refine(b)
    assert int(new_board.get(6 + HIDDEN_ROWS, 1)) == COLOR_OJAMA
