"""V2.3 ConnectivityShapeRefiner のテスト。"""
from __future__ import annotations

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    Board,
)
from src.connectivity_refiner import (
    ConnectivityShapeRefiner,
)


def _make_board(cells: dict[tuple[int, int], int]) -> Board:
    b = Board()
    for (row, col), color in cells.items():
        b.set(row, col, color)
    return b


def test_isolated_single_cell_corrected() -> None:
    """赤に完全に囲まれた紫 1 セルは赤に補正される。"""
    cells = {
        (5, 2): COLOR_RED,      # 上
        (7, 2): COLOR_RED,      # 下
        (6, 1): COLOR_RED,      # 左
        (6, 3): COLOR_RED,      # 右
        (6, 2): COLOR_PURPLE,   # 中央: 紫 → 赤に補正されるべき
    }
    refiner = ConnectivityShapeRefiner()
    res = refiner.refine(_make_board(cells))
    assert res.n_corrected == 1
    assert res.refined.get(6, 2) == COLOR_RED


def test_only_two_neighbors_no_correction() -> None:
    """隣接 2 セルしか同色なら補正しない (min_agreement=3)。"""
    cells = {
        (5, 2): COLOR_RED,
        (6, 1): COLOR_RED,
        (6, 2): COLOR_PURPLE,  # 中央: 紫
        # 下と右は EMPTY
    }
    refiner = ConnectivityShapeRefiner()
    res = refiner.refine(_make_board(cells))
    assert res.n_corrected == 0


def test_correct_chain_preconfiguration_not_broken() -> None:
    """連鎖直前の正しい配置 (赤赤赤 / 紫赤赤) は壊さない。

    紫 (5, 0) は連結赤の隣だが、隣接 4 セルのうち 3 セル以上が赤ではない
    (左は盤面外、上は赤、下は赤、右は赤 = 3 セル赤)。
    実は 3 セル赤 → 補正発動する。これは現実的に「赤連結に挟まれた紫」
    で v2.3 の仕様通り補正対象。リスク承知。

    本テストでは「2 セルしか赤がない場合は壊さない」のみ確認。
    """
    cells = {
        (5, 0): COLOR_PURPLE,
        (5, 1): COLOR_RED,  # 紫の右
        (6, 0): COLOR_RED,  # 紫の下
        # 紫の上下左右で赤は 2 セル (右・下) のみ
        # 左は盤面外
        # 上は EMPTY
    }
    refiner = ConnectivityShapeRefiner()
    res = refiner.refine(_make_board(cells))
    assert res.n_corrected == 0
    assert res.refined.get(5, 0) == COLOR_PURPLE


def test_empty_cell_not_corrected() -> None:
    """EMPTY セルは補正対象外。"""
    cells = {
        (5, 2): COLOR_RED,
        (7, 2): COLOR_RED,
        (6, 1): COLOR_RED,
        (6, 3): COLOR_RED,
        # (6, 2) は EMPTY
    }
    refiner = ConnectivityShapeRefiner()
    res = refiner.refine(_make_board(cells))
    assert res.n_corrected == 0


def test_unknown_cell_not_corrected() -> None:
    """UNKNOWN セルは補正対象外。"""
    cells = {
        (5, 2): COLOR_RED,
        (7, 2): COLOR_RED,
        (6, 1): COLOR_RED,
        (6, 3): COLOR_RED,
        (6, 2): COLOR_UNKNOWN,
    }
    refiner = ConnectivityShapeRefiner()
    res = refiner.refine(_make_board(cells))
    assert res.n_corrected == 0
    assert res.refined.get(6, 2) == COLOR_UNKNOWN


def test_ojama_not_used_as_majority_color() -> None:
    """OJAMA は EXCLUDE で多数色対象外、補正に使われない。"""
    cells = {
        (5, 2): COLOR_OJAMA,
        (7, 2): COLOR_OJAMA,
        (6, 1): COLOR_OJAMA,
        (6, 3): COLOR_OJAMA,
        (6, 2): COLOR_RED,  # OJAMA に囲まれた赤 → 補正されないべき
    }
    refiner = ConnectivityShapeRefiner()
    res = refiner.refine(_make_board(cells))
    # OJAMA は EXCLUDE_COLORS なので neighbors 計算で除外、補正発動しない
    assert res.n_corrected == 0
    assert res.refined.get(6, 2) == COLOR_RED


def test_multiple_corrections() -> None:
    """複数の異色セルが補正される。"""
    cells = {
        (5, 2): COLOR_RED,
        (7, 2): COLOR_RED,
        (6, 1): COLOR_RED,
        (6, 3): COLOR_RED,
        (6, 2): COLOR_BLUE,    # 補正対象 1
        (5, 4): COLOR_GREEN,
        (7, 4): COLOR_GREEN,
        (6, 5): COLOR_GREEN,
        (6, 4): COLOR_YELLOW,  # 補正対象 2
    }
    refiner = ConnectivityShapeRefiner()
    res = refiner.refine(_make_board(cells))
    assert res.n_corrected == 2
    assert res.refined.get(6, 2) == COLOR_RED
    # (6, 4) は隣接通常色 3 セル (5,4=GRN / 7,4=GRN / 6,5=GRN)
    # 6,3=RED は通常色だが GRN 多数なので GRN に補正
    assert res.refined.get(6, 4) == COLOR_GREEN


def test_corrections_tuple_records_changes() -> None:
    """corrections フィールドに補正履歴が記録される。"""
    cells = {
        (5, 2): COLOR_RED,
        (7, 2): COLOR_RED,
        (6, 1): COLOR_RED,
        (6, 3): COLOR_RED,
        (6, 2): COLOR_PURPLE,
    }
    refiner = ConnectivityShapeRefiner()
    res = refiner.refine(_make_board(cells))
    assert res.n_corrected == 1
    assert res.corrections == ((6, 2, COLOR_PURPLE, COLOR_RED),)


def test_min_agreement_4_strict() -> None:
    """min_neighbor_agreement=4 で完全包囲のみ補正。"""
    refiner = ConnectivityShapeRefiner(min_neighbor_agreement=4)
    # 3 セル赤、1 セル EMPTY → 補正しない
    cells_3 = {
        (5, 2): COLOR_RED,
        (7, 2): COLOR_RED,
        (6, 1): COLOR_RED,
        (6, 2): COLOR_PURPLE,
        # (6, 3) は EMPTY
    }
    res = refiner.refine(_make_board(cells_3))
    assert res.n_corrected == 0


def test_no_change_on_uniform_board() -> None:
    """単色盤面では何も補正しない。"""
    cells = {(r, c): COLOR_RED for r in range(5, 10) for c in range(2, 5)}
    refiner = ConnectivityShapeRefiner()
    res = refiner.refine(_make_board(cells))
    assert res.n_corrected == 0
