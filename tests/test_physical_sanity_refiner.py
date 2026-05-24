"""W6 PhysicalSanityRefiner のテスト。"""
from __future__ import annotations

from src.board import (
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_RED,
    COLOR_UNKNOWN,
    Board,
)
from src.physical_sanity_refiner import (
    PhysicalSanityRefiner,
    SanityRefineResult,
)


def _make_board(cells: dict[tuple[int, int], int]) -> Board:
    b = Board()
    for (r, c), color in cells.items():
        b.set(r, c, color)
    return b


def test_no_violation_no_correction() -> None:
    """4 連結なし → 補正なし。"""
    cells = {
        (12, 0): COLOR_RED, (12, 1): COLOR_RED, (12, 2): COLOR_RED,
        # 3 連結
    }
    refiner = PhysicalSanityRefiner()
    res = refiner.refine(_make_board(cells))
    assert res.n_corrected == 0
    assert res.refined.get(12, 0) == COLOR_RED


def test_horizontal_4_link_corrected() -> None:
    """横 4 連結 → 1 セル UNKNOWN 化。"""
    cells = {
        (12, 0): COLOR_RED, (12, 1): COLOR_RED,
        (12, 2): COLOR_RED, (12, 3): COLOR_RED,
    }
    refiner = PhysicalSanityRefiner()
    res = refiner.refine(_make_board(cells))
    assert res.n_corrected == 1
    # 4 セル中 1 つが UNKNOWN
    n_unknown = sum(
        1 for c in [(12, 0), (12, 1), (12, 2), (12, 3)]
        if res.refined.get(*c) == COLOR_UNKNOWN
    )
    n_red = sum(
        1 for c in [(12, 0), (12, 1), (12, 2), (12, 3)]
        if res.refined.get(*c) == COLOR_RED
    )
    assert n_unknown == 1
    assert n_red == 3


def test_vertical_4_link_corrected() -> None:
    cells = {
        (9, 0): COLOR_BLUE, (10, 0): COLOR_BLUE,
        (11, 0): COLOR_BLUE, (12, 0): COLOR_BLUE,
    }
    refiner = PhysicalSanityRefiner()
    res = refiner.refine(_make_board(cells))
    assert res.n_corrected == 1


def test_5_link_corrected_to_4() -> None:
    """5 連結も 1 セル UNKNOWN 化、結果 4 セル残る (まだ違反だが、
    refine 1 回では 1 グループ 1 セル補正)。"""
    cells = {(12, c): COLOR_RED for c in range(5)}
    refiner = PhysicalSanityRefiner()
    res = refiner.refine(_make_board(cells))
    assert res.n_corrected == 1


def test_corner_cell_picked_first() -> None:
    """L 字 4 連結 → 最外周 (隣接 1) のセルが選ばれる。

    配置:
        (10,0) RED
        (11,0) RED
        (12,0) RED (角、連鎖頭)
        (12,1) RED (連鎖尾)
    隣接数: (10,0)=1, (11,0)=2, (12,0)=2, (12,1)=1
    最外周 = (10,0) または (12,1)、最上段優先で (10,0)
    """
    cells = {
        (10, 0): COLOR_RED,
        (11, 0): COLOR_RED,
        (12, 0): COLOR_RED,
        (12, 1): COLOR_RED,
    }
    refiner = PhysicalSanityRefiner()
    res = refiner.refine(_make_board(cells))
    assert res.n_corrected == 1
    # (10, 0) が UNKNOWN になるはず (隣接 1 で最上段)
    assert res.refined.get(10, 0) == COLOR_UNKNOWN


def test_ojama_4_link_not_corrected() -> None:
    """お邪魔 4+ 連結は連鎖しないので補正しない。"""
    cells = {(12, c): COLOR_OJAMA for c in range(4)}
    refiner = PhysicalSanityRefiner()
    res = refiner.refine(_make_board(cells))
    assert res.n_corrected == 0


def test_multiple_groups_corrected() -> None:
    """複数の 4+ 連結があれば全部補正 (各 1 セルずつ)。"""
    cells = {
        (12, 0): COLOR_RED, (12, 1): COLOR_RED,
        (12, 2): COLOR_RED, (12, 3): COLOR_RED,
        (10, 4): COLOR_BLUE, (10, 5): COLOR_BLUE,
        (11, 4): COLOR_BLUE, (11, 5): COLOR_BLUE,
    }
    refiner = PhysicalSanityRefiner()
    res = refiner.refine(_make_board(cells))
    assert res.n_corrected == 2


def test_correction_metadata() -> None:
    """corrections フィールドに古い色・新しい色が記録される。"""
    cells = {(12, c): COLOR_RED for c in range(4)}
    refiner = PhysicalSanityRefiner()
    res = refiner.refine(_make_board(cells))
    assert len(res.corrections) == 1
    row, col, old, new = res.corrections[0]
    assert old == COLOR_RED
    assert new == COLOR_UNKNOWN
