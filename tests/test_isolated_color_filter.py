"""W7-N3 IsolatedColorFilter のテスト。"""
from __future__ import annotations

from src.board import (
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    Board,
)
from src.isolated_color_filter import IsolatedColorFilter


def _board(cells: dict[tuple[int, int], int]) -> Board:
    b = Board()
    for (r, c), color in cells.items():
        b.set(r, c, color)
    return b


def test_isolated_yellow_in_red_corrected() -> None:
    """周囲が全部赤の中の黄 1 セル → UNKNOWN。"""
    cells = {
        (5, 2): COLOR_RED,
        (7, 2): COLOR_RED,
        (6, 1): COLOR_RED,
        (6, 3): COLOR_RED,
        (6, 2): COLOR_YELLOW,
    }
    flt = IsolatedColorFilter()
    res = flt.refine(_board(cells))
    assert res.n_corrected == 1
    assert res.refined.get(6, 2) == COLOR_UNKNOWN


def test_same_color_not_corrected() -> None:
    """周囲全部同色 → そのまま。"""
    cells = {(r, c): COLOR_RED for r in range(5, 8) for c in range(1, 4)}
    flt = IsolatedColorFilter()
    res = flt.refine(_board(cells))
    assert res.n_corrected == 0


def test_empty_neighbor_does_not_count() -> None:
    """EMPTY 隣接は別色カウントしない。"""
    cells = {
        (6, 2): COLOR_YELLOW,
        # 周囲は全 EMPTY
    }
    flt = IsolatedColorFilter()
    res = flt.refine(_board(cells))
    assert res.n_corrected == 0
    assert res.refined.get(6, 2) == COLOR_YELLOW


def test_ojama_neighbor_does_not_count() -> None:
    """OJAMA 隣接も別色カウントしない (NORMAL_COLORS 外)。"""
    cells = {
        (5, 2): COLOR_OJAMA,
        (7, 2): COLOR_OJAMA,
        (6, 1): COLOR_OJAMA,
        (6, 3): COLOR_OJAMA,
        (6, 2): COLOR_YELLOW,
    }
    flt = IsolatedColorFilter()
    res = flt.refine(_board(cells))
    assert res.n_corrected == 0


def test_two_different_neighbors_not_enough() -> None:
    """別色隣接 2 つだけ → min_different=3 で補正しない。"""
    cells = {
        (5, 2): COLOR_RED,
        (6, 1): COLOR_RED,
        (6, 2): COLOR_YELLOW,
        # (7, 2), (6, 3) は EMPTY
    }
    flt = IsolatedColorFilter()
    res = flt.refine(_board(cells))
    assert res.n_corrected == 0


def test_multiple_isolated_corrected() -> None:
    """複数の孤立色を全て補正。"""
    cells = {
        (5, 2): COLOR_RED, (7, 2): COLOR_RED,
        (6, 1): COLOR_RED, (6, 3): COLOR_RED,
        (6, 2): COLOR_YELLOW,  # 補正対象 1
        # 別グループ
        (5, 5): COLOR_BLUE, (7, 5): COLOR_BLUE,
        (6, 4): COLOR_BLUE,
        (6, 5): COLOR_RED,  # 補正対象 2 (周囲 BLUE 3)
    }
    flt = IsolatedColorFilter()
    res = flt.refine(_board(cells))
    assert res.n_corrected == 2


def test_min_different_4_strict() -> None:
    """min_different=4 で完全包囲のみ補正。"""
    flt = IsolatedColorFilter(min_different_neighbors=4)
    cells_3 = {
        (5, 2): COLOR_RED,
        (7, 2): COLOR_RED,
        (6, 1): COLOR_RED,
        (6, 2): COLOR_YELLOW,
        # (6, 3) は EMPTY
    }
    res = flt.refine(_board(cells_3))
    # 別色隣接 3 のみ → min 4 で補正しない
    assert res.n_corrected == 0
