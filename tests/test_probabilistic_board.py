"""W3.0 ProbabilisticBoard / ProbabilisticCell のテスト。"""
from __future__ import annotations

import numpy as np

from src.board import (
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_RED,
    COLOR_UNKNOWN,
    Board,
)
from src.probabilistic_board import (
    CERTAIN_THRESHOLD,
    ProbabilisticBoard,
    ProbabilisticCell,
)


def test_cell_certain() -> None:
    """certain factory は単一色 1.0。"""
    cell = ProbabilisticCell.certain(COLOR_RED)
    assert cell.most_likely() == (COLOR_RED, 1.0)
    assert cell.is_certain()


def test_cell_uniform_not_certain() -> None:
    """uniform は確率均等、確定ではない。"""
    cell = ProbabilisticCell.uniform()
    assert not cell.is_certain()
    # 均等分布のエントロピーは log(7) ≈ 1.95
    assert 1.9 < cell.entropy() < 2.0


def test_cell_normalize() -> None:
    """probs の合計が 1.0 に正規化される。"""
    cell = ProbabilisticCell(probs={COLOR_RED: 2.0, COLOR_BLUE: 8.0})
    cell.normalize()
    assert cell.probs[COLOR_RED] == 0.2
    assert cell.probs[COLOR_BLUE] == 0.8


def test_cell_most_likely_with_distribution() -> None:
    cell = ProbabilisticCell(
        probs={COLOR_RED: 0.6, COLOR_BLUE: 0.4},
    )
    color, prob = cell.most_likely()
    assert color == COLOR_RED
    assert prob == 0.6


def test_cell_is_certain_threshold() -> None:
    cell = ProbabilisticCell(probs={COLOR_RED: 0.96, COLOR_BLUE: 0.04})
    assert cell.is_certain()
    cell2 = ProbabilisticCell(probs={COLOR_RED: 0.50, COLOR_BLUE: 0.50})
    assert not cell2.is_certain()


def test_board_from_board() -> None:
    """既存 Board → ProbabilisticBoard 変換。"""
    b = Board()
    b.set(11, 2, COLOR_RED)
    b.set(12, 2, COLOR_BLUE)
    pb = ProbabilisticBoard.from_board(b)
    assert pb.cell(11, 2).most_likely() == (COLOR_RED, 1.0)
    assert pb.cell(12, 2).most_likely() == (COLOR_BLUE, 1.0)
    # 他は EMPTY
    assert pb.cell(11, 0).most_likely() == (COLOR_EMPTY, 1.0)


def test_board_unknown_becomes_uniform() -> None:
    """UNKNOWN セルは均等分布になる。"""
    b = Board()
    b.set(0, 2, COLOR_UNKNOWN)
    pb = ProbabilisticBoard.from_board(b)
    cell = pb.cell(0, 2)
    assert not cell.is_certain()
    # 均等分布のはず
    assert all(0.1 < p < 0.2 for p in cell.probs.values())


def test_board_to_board_certain_only() -> None:
    """to_board: 確定セルは確定色、不確定は UNKNOWN。"""
    pb = ProbabilisticBoard()
    pb.set_certain(11, 2, COLOR_RED)
    pb.set_distribution(0, 3, {COLOR_RED: 0.6, COLOR_BLUE: 0.4})
    board = pb.to_board()
    assert board.get(11, 2) == COLOR_RED
    # 60/40 は CERTAIN_THRESHOLD 0.95 未満 → UNKNOWN
    assert board.get(0, 3) == COLOR_UNKNOWN


def test_board_to_board_high_confidence_passes() -> None:
    """0.96 等の高確率は確定として通す。"""
    pb = ProbabilisticBoard()
    pb.set_distribution(0, 3, {COLOR_RED: 0.96, COLOR_BLUE: 0.04})
    board = pb.to_board()
    assert board.get(0, 3) == COLOR_RED


def test_board_n_uncertain() -> None:
    """不確定セル数のカウント。"""
    pb = ProbabilisticBoard()
    pb.set_distribution(0, 3, {COLOR_RED: 0.5, COLOR_BLUE: 0.5})
    pb.set_distribution(0, 4, {COLOR_RED: 0.6, COLOR_BLUE: 0.4})
    # 全 13×6=78 セル中、上記 2 セルが不確定
    assert pb.n_uncertain() == 2


def test_board_total_uncertainty_zero_for_certain() -> None:
    """全セル確定なら全エントロピー 0。"""
    pb = ProbabilisticBoard()  # 全 EMPTY 確定
    assert pb.total_uncertainty() == 0.0
