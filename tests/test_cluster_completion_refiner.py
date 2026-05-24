"""ClusterCompletionRefiner テスト。"""
from __future__ import annotations

import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, Board, COLOR_BLUE, COLOR_EMPTY,
    COLOR_GREEN, COLOR_RED, HIDDEN_ROWS,
)
from src.cluster_completion_refiner import ClusterCompletionRefiner


def _set(b: Board, vrow: int, col: int, color: int) -> None:
    b.set(vrow + HIDDEN_ROWS, col, color)


def test_3_cluster_with_isolated_diff_color() -> None:
    """3 連結 + 隣接 1 cell 異色 → 補正。"""
    b = Board()
    # 横 3 連結 (赤): r10 c0,c1,c2
    _set(b, 10, 0, COLOR_RED)
    _set(b, 10, 1, COLOR_RED)
    _set(b, 10, 2, COLOR_RED)
    # 隣接異色 (青): r10 c3
    _set(b, 10, 3, COLOR_BLUE)
    refiner = ClusterCompletionRefiner()
    new_board, mask = refiner.refine(b)
    # c3 は RED に補正される
    assert int(new_board.get(10 + HIDDEN_ROWS, 3)) == COLOR_RED
    assert mask[10 + HIDDEN_ROWS, 3]


def test_no_refine_when_no_3_cluster() -> None:
    """3 cluster がない → 補正なし。"""
    b = Board()
    _set(b, 10, 0, COLOR_RED)
    _set(b, 10, 1, COLOR_RED)
    _set(b, 10, 2, COLOR_BLUE)
    refiner = ClusterCompletionRefiner()
    new_board, mask = refiner.refine(b)
    assert not mask.any()


def test_skip_during_chain() -> None:
    b = Board()
    _set(b, 10, 0, COLOR_RED)
    _set(b, 10, 1, COLOR_RED)
    _set(b, 10, 2, COLOR_RED)
    _set(b, 10, 3, COLOR_BLUE)
    refiner = ClusterCompletionRefiner()
    new_board, mask = refiner.refine(b, is_chain=True)
    assert int(new_board.get(10 + HIDDEN_ROWS, 3)) == COLOR_BLUE
    assert not mask.any()


def test_em_neighbor_not_modified() -> None:
    """EM 隣接 cell は対象外 (補正しない)。"""
    b = Board()
    _set(b, 10, 0, COLOR_RED)
    _set(b, 10, 1, COLOR_RED)
    _set(b, 10, 2, COLOR_RED)
    # c3 は EM (デフォルト)
    refiner = ClusterCompletionRefiner()
    new_board, mask = refiner.refine(b)
    assert int(new_board.get(10 + HIDDEN_ROWS, 3)) == COLOR_EMPTY
    assert not mask.any()
