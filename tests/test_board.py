"""
board.py のテスト
"""

import json

import numpy as np
import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_RED,
    COLOR_YELLOW,
    Board,
)


# ============================
# ヘルパー
# ============================

def empty_grid() -> list[list[int]]:
    """13行×6列の全空グリッドを生成する。"""
    return [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]


def make_board_with_column(col: int, heights: list[int]) -> Board:
    """
    指定列に下から積み上げたBoardを生成する。
    heights[i] = その列の高さ
    """
    grid = empty_grid()
    for c, h in enumerate(heights):
        for r in range(h):
            grid[BOARD_ROWS - 1 - r][c] = COLOR_RED
    return Board.from_list(grid)


# ============================
# Board生成テスト
# ============================

class TestBoardCreation:
    def test_empty_board_is_all_zeros(self):
        board = Board()
        assert board.count_puyos() == 0

    def test_from_list_basic(self):
        grid = empty_grid()
        grid[12][0] = COLOR_RED
        board = Board.from_list(grid)
        assert board.get(12, 0) == COLOR_RED

    def test_from_list_wrong_rows(self):
        with pytest.raises(ValueError, match="行数"):
            Board.from_list([[0] * BOARD_COLS] * 5)  # 5行はNG

    def test_from_list_wrong_cols(self):
        grid = empty_grid()
        grid[0] = [0] * 3  # 3列はNG
        with pytest.raises(ValueError, match="列数"):
            Board.from_list(grid)

    def test_from_list_invalid_color(self):
        grid = empty_grid()
        grid[0][0] = 7  # 7は無効
        with pytest.raises(ValueError, match="無効な色値"):
            Board.from_list(grid)

    def test_ojama_is_valid(self):
        grid = empty_grid()
        grid[12][5] = COLOR_OJAMA
        board = Board.from_list(grid)
        assert board.get(12, 5) == COLOR_OJAMA


# ============================
# get / set テスト
# ============================

class TestGetSet:
    def test_set_and_get(self):
        board = Board()
        board.set(5, 3, COLOR_BLUE)
        assert board.get(5, 3) == COLOR_BLUE

    def test_set_invalid_color(self):
        board = Board()
        with pytest.raises(ValueError):
            board.set(0, 0, 8)  # 8は無効

    def test_get_out_of_range_row(self):
        board = Board()
        with pytest.raises(ValueError):
            board.get(13, 0)

    def test_get_out_of_range_col(self):
        board = Board()
        with pytest.raises(ValueError):
            board.get(0, 6)

    def test_set_out_of_range(self):
        board = Board()
        with pytest.raises(ValueError):
            board.set(-1, 0, COLOR_RED)


# ============================
# height_of テスト
# ============================

class TestHeightOf:
    def test_empty_column_height_is_zero(self):
        board = Board()
        assert board.height_of(0) == 0

    def test_single_puyo_at_bottom(self):
        grid = empty_grid()
        grid[12][0] = COLOR_RED
        board = Board.from_list(grid)
        assert board.height_of(0) == 1

    def test_full_column_height(self):
        grid = empty_grid()
        for r in range(BOARD_ROWS):
            grid[r][2] = COLOR_GREEN
        board = Board.from_list(grid)
        assert board.height_of(2) == BOARD_ROWS

    def test_height_3(self):
        board = make_board_with_column(0, [3, 0, 0, 0, 0, 0])
        assert board.height_of(0) == 3

    def test_invalid_col(self):
        board = Board()
        with pytest.raises(ValueError):
            board.height_of(6)


# ============================
# is_dead テスト
# ============================

class TestIsDead:
    def test_empty_board_not_dead(self):
        board = Board()
        assert board.is_dead() is False

    def test_dead_when_death_col_top_occupied(self):
        grid = empty_grid()
        grid[0][2] = COLOR_RED  # 3列目(index:2)の最上段
        board = Board.from_list(grid)
        assert board.is_dead() is True

    def test_not_dead_other_col_top(self):
        grid = empty_grid()
        grid[0][1] = COLOR_RED  # 2列目は死なない
        board = Board.from_list(grid)
        assert board.is_dead() is False

    def test_not_dead_death_col_not_top(self):
        grid = empty_grid()
        grid[1][2] = COLOR_RED  # 3列目だが最上段ではない
        board = Board.from_list(grid)
        assert board.is_dead() is False


# ============================
# count_puyos テスト
# ============================

class TestCountPuyos:
    def test_empty_board_count_zero(self):
        board = Board()
        assert board.count_puyos() == 0

    def test_count_with_puyos(self):
        grid = empty_grid()
        grid[12][0] = COLOR_RED
        grid[12][1] = COLOR_BLUE
        grid[11][0] = COLOR_GREEN
        board = Board.from_list(grid)
        assert board.count_puyos() == 3

    def test_ojama_counted(self):
        grid = empty_grid()
        grid[12][0] = COLOR_OJAMA
        board = Board.from_list(grid)
        assert board.count_puyos() == 1

    def test_full_board_count(self):
        grid = [[COLOR_RED] * BOARD_COLS for _ in range(BOARD_ROWS)]
        board = Board.from_list(grid)
        assert board.count_puyos() == BOARD_ROWS * BOARD_COLS


# ============================
# copy テスト
# ============================

class TestCopy:
    def test_copy_is_equal(self):
        grid = empty_grid()
        grid[12][0] = COLOR_YELLOW
        board = Board.from_list(grid)
        copied = board.copy()
        assert board == copied

    def test_copy_is_independent(self):
        board = Board()
        board.set(0, 0, COLOR_RED)
        copied = board.copy()
        copied.set(0, 0, COLOR_BLUE)
        assert board.get(0, 0) == COLOR_RED  # 元は変わらない


# ============================
# to_dict / from_dict テスト
# ============================

class TestSerialization:
    def test_to_dict_has_grid(self):
        board = Board()
        d = board.to_dict()
        assert "grid" in d
        assert len(d["grid"]) == BOARD_ROWS
        assert len(d["grid"][0]) == BOARD_COLS

    def test_round_trip(self):
        grid = empty_grid()
        grid[5][3] = COLOR_PURPLE = 5
        board = Board.from_list(grid)
        restored = Board.from_dict(board.to_dict())
        assert board == restored

    def test_to_json_parseable(self):
        board = Board()
        board.set(12, 0, COLOR_RED)
        j = board.to_json()
        parsed = json.loads(j)
        assert parsed["grid"][12][0] == COLOR_RED


# ============================
# __eq__ テスト
# ============================

class TestEquality:
    def test_two_empty_boards_equal(self):
        assert Board() == Board()

    def test_different_boards_not_equal(self):
        b1 = Board()
        b2 = Board()
        b2.set(0, 0, COLOR_RED)
        assert b1 != b2

    def test_eq_with_non_board(self):
        assert Board().__eq__("not a board") == NotImplemented
