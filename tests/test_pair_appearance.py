"""V2.2 PairAppearanceConsistency のテスト。"""
from __future__ import annotations

from src.board import (
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_RED,
    Board,
)
from src.pair_appearance import (
    PairAppearanceConsistency,
    PairConsistencyResult,
)


def _make_board(cells: dict[tuple[int, int], int]) -> Board:
    b = Board()
    for (row, col), color in cells.items():
        b.set(row, col, color)
    return b


def test_no_new_cells_consistent() -> None:
    """変化なしは整合あり。"""
    cells = {(12, 2): COLOR_RED}
    pac = PairAppearanceConsistency()
    res = pac.check(_make_board(cells), _make_board(cells))
    assert res.is_consistent is True
    assert res.n_new_cells == 0


def test_two_new_cells_consistent() -> None:
    """2 セル新規 = 整合 (ペア)。"""
    prev = _make_board({})
    cur = _make_board({(11, 2): COLOR_RED, (12, 2): COLOR_BLUE})
    pac = PairAppearanceConsistency()
    res = pac.check(prev, cur)
    assert res.is_consistent is True
    assert res.n_new_cells == 2


def test_one_new_cell_inconsistent_with_candidates() -> None:
    """1 セル新規 = 不整合、相方位置候補が返る。"""
    prev = _make_board({})
    cur = _make_board({(12, 2): COLOR_RED})
    pac = PairAppearanceConsistency()
    res = pac.check(prev, cur)
    assert res.is_consistent is False
    assert res.n_new_cells == 1
    # 候補: (11, 2)=上、(12, 1)=左、(12, 3)=右
    assert (11, 2) in res.candidate_partner_positions
    assert (12, 1) in res.candidate_partner_positions
    assert (12, 3) in res.candidate_partner_positions


def test_three_new_cells_inconsistent_no_candidates() -> None:
    """3 セル新規 = 不整合、候補は空。"""
    prev = _make_board({})
    cur = _make_board({
        (10, 2): COLOR_RED,
        (11, 2): COLOR_BLUE,
        (12, 2): COLOR_RED,
    })
    pac = PairAppearanceConsistency()
    res = pac.check(prev, cur)
    assert res.is_consistent is False
    assert res.n_new_cells == 3
    assert res.candidate_partner_positions == ()


def test_partner_candidates_filter_occupied() -> None:
    """相方候補は prev でも cur でも EMPTY のセルのみ。"""
    prev = _make_board({(11, 2): COLOR_BLUE})  # 上はもう既に占有
    cur = _make_board({(11, 2): COLOR_BLUE, (12, 2): COLOR_RED})
    pac = PairAppearanceConsistency()
    res = pac.check(prev, cur)
    # 新規は (12, 2) のみ
    assert res.n_new_cells == 1
    # (11, 2) は prev で BLUE 占有 → 候補外
    assert (11, 2) not in res.candidate_partner_positions
    # (12, 1) と (12, 3) は EMPTY → 候補
    assert (12, 1) in res.candidate_partner_positions
    assert (12, 3) in res.candidate_partner_positions


def test_corner_cell_limited_candidates() -> None:
    """端のセル (row=12, col=0) は候補が制限される。"""
    prev = _make_board({})
    cur = _make_board({(12, 0): COLOR_RED})
    pac = PairAppearanceConsistency()
    res = pac.check(prev, cur)
    # (12, 0) の候補: (11, 0)=上、(12, 1)=右 (左は盤面外)
    assert (11, 0) in res.candidate_partner_positions
    assert (12, 1) in res.candidate_partner_positions
    assert len(res.candidate_partner_positions) == 2
