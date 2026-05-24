"""board_rules.apply_gravity のテスト。"""
from __future__ import annotations

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_RED,
    COLOR_UNKNOWN,
    Board,
)
from src.board_rules import apply_gravity, clear_floating_above_gap, diff_boards


def _empty_grid() -> list[list[int]]:
    return [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]


def test_gravity_drops_single_floating_puyo() -> None:
    grid = _empty_grid()
    grid[5][3] = COLOR_GREEN  # row 5 に浮いている緑
    before = Board.from_list(grid)

    after = apply_gravity(before)

    assert after.get(12, 3) == COLOR_GREEN
    assert after.get(5, 3) == COLOR_EMPTY


def test_gravity_preserves_bottom_packed_column() -> None:
    grid = _empty_grid()
    grid[12][0] = COLOR_RED
    grid[11][0] = COLOR_BLUE
    grid[10][0] = COLOR_GREEN
    before = Board.from_list(grid)

    after = apply_gravity(before)

    assert after.get(12, 0) == COLOR_RED
    assert after.get(11, 0) == COLOR_BLUE
    assert after.get(10, 0) == COLOR_GREEN


def test_gravity_preserves_color_order() -> None:
    grid = _empty_grid()
    grid[4][1] = COLOR_RED      # 上
    grid[8][1] = COLOR_BLUE     # 中
    grid[12][1] = COLOR_GREEN   # 下（既に接地）
    before = Board.from_list(grid)

    after = apply_gravity(before)

    # 下から赤(元上)→青(元中)→緑(元下)... ではなく、
    # 相対順序保持のため下から緑, 青, 赤 となる
    assert after.get(12, 1) == COLOR_GREEN
    assert after.get(11, 1) == COLOR_BLUE
    assert after.get(10, 1) == COLOR_RED
    assert after.get(9, 1) == COLOR_EMPTY


def test_gravity_handles_ojama_like_colored_puyo() -> None:
    grid = _empty_grid()
    grid[3][2] = COLOR_OJAMA
    grid[12][2] = COLOR_RED
    before = Board.from_list(grid)

    after = apply_gravity(before)

    # おじゃまも重力対象
    assert after.get(12, 2) == COLOR_RED
    assert after.get(11, 2) == COLOR_OJAMA


def test_gravity_skips_hidden_row_by_default() -> None:
    grid = _empty_grid()
    grid[0][4] = COLOR_GREEN  # 隠し段（row 0）
    before = Board.from_list(grid)

    after = apply_gravity(before, skip_hidden=True)

    # 隠し段は補正対象外なので動かない
    assert after.get(0, 4) == COLOR_GREEN
    assert after.get(12, 4) == COLOR_EMPTY


def test_gravity_preserves_unknown_position() -> None:
    grid = _empty_grid()
    grid[10][5] = COLOR_UNKNOWN
    grid[4][5] = COLOR_BLUE
    before = Board.from_list(grid)

    after = apply_gravity(before)

    # UNKNOWN は位置固定
    assert after.get(10, 5) == COLOR_UNKNOWN
    # BLUE は UNKNOWN の下ではなく上に詰まる（下段は UNKNOWN でブロック）
    assert after.get(12, 5) == COLOR_BLUE
    assert after.get(4, 5) == COLOR_EMPTY


def test_diff_boards_empty_when_same() -> None:
    b1 = Board.from_list(_empty_grid())
    b2 = Board.from_list(_empty_grid())
    assert diff_boards(b1, b2) == []


def test_clear_floating_above_gap_removes_isolated_top_cell() -> None:
    """× マーク相当の r01 孤立セルが消される（接地スタックは保持）。"""
    grid = _empty_grid()
    grid[1][2] = COLOR_RED  # × 相当の最上段孤立
    # 下部に接地スタック r10-r12
    grid[10][2] = COLOR_BLUE
    grid[11][2] = COLOR_GREEN
    grid[12][2] = COLOR_RED
    before = Board.from_list(grid)

    after = clear_floating_above_gap(before, min_gap=2)

    assert after.get(1, 2) == COLOR_EMPTY       # 浮遊赤は消去
    assert after.get(10, 2) == COLOR_BLUE        # スタックは保持
    assert after.get(11, 2) == COLOR_GREEN
    assert after.get(12, 2) == COLOR_RED


def test_clear_floating_above_gap_preserves_contiguous_tall_stack() -> None:
    """浮遊ではない連続スタックは保持する。"""
    grid = _empty_grid()
    for row in range(2, 13):
        grid[row][0] = COLOR_RED
    before = Board.from_list(grid)

    after = clear_floating_above_gap(before, min_gap=2)

    for row in range(2, 13):
        assert after.get(row, 0) == COLOR_RED


def test_clear_floating_above_gap_keeps_small_gap() -> None:
    """接地スタックから 1 行だけ離れた上部セルは保持（gap < min_gap）。"""
    grid = _empty_grid()
    # 接地スタック r09-r12（連続）
    grid[9][1] = COLOR_BLUE
    grid[10][1] = COLOR_GREEN
    grid[11][1] = COLOR_BLUE
    grid[12][1] = COLOR_RED
    # gap=1 の上部セル
    grid[7][1] = COLOR_GREEN
    before = Board.from_list(grid)

    after = clear_floating_above_gap(before, min_gap=2)

    assert after.get(7, 1) == COLOR_GREEN        # gap=1 なので保持
    assert after.get(9, 1) == COLOR_BLUE
    assert after.get(12, 1) == COLOR_RED


def test_clear_floating_above_gap_clears_multiple_top_cells() -> None:
    """浮遊セルが複数あれば全部消す。"""
    grid = _empty_grid()
    grid[1][3] = COLOR_RED
    grid[2][3] = COLOR_RED
    grid[3][3] = COLOR_RED
    # スタック
    grid[10][3] = COLOR_BLUE
    grid[11][3] = COLOR_BLUE
    grid[12][3] = COLOR_BLUE
    before = Board.from_list(grid)

    after = clear_floating_above_gap(before, min_gap=2)

    assert after.get(1, 3) == COLOR_EMPTY
    assert after.get(2, 3) == COLOR_EMPTY
    assert after.get(3, 3) == COLOR_EMPTY
    assert after.get(10, 3) == COLOR_BLUE
    assert after.get(11, 3) == COLOR_BLUE
    assert after.get(12, 3) == COLOR_BLUE


def test_clear_floating_above_gap_empty_column_unchanged() -> None:
    """空列は変化なし。"""
    before = Board.from_list(_empty_grid())
    after = clear_floating_above_gap(before, min_gap=2)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            assert after.get(row, col) == COLOR_EMPTY


def test_diff_boards_lists_changes() -> None:
    g1 = _empty_grid()
    g1[5][3] = COLOR_GREEN
    g2 = _empty_grid()
    g2[12][3] = COLOR_GREEN
    b1 = Board.from_list(g1)
    b2 = Board.from_list(g2)

    changes = diff_boards(b1, b2)

    assert len(changes) == 2
    rows = {(c.row, c.col, c.before, c.after) for c in changes}
    assert (5, 3, COLOR_GREEN, COLOR_EMPTY) in rows
    assert (12, 3, COLOR_EMPTY, COLOR_GREEN) in rows
