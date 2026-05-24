"""src/all_clear_detector.py のテスト (Phase R)。"""
from __future__ import annotations

import pytest

from src.all_clear_detector import (
    AllClearResult,
    count_puyos_on_field,
    is_all_clear,
)
from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_RED,
    Board,
)


def _empty_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(grid)


def _board_with_color() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    grid[BOARD_ROWS - 1][0] = COLOR_RED
    return Board.from_list(grid)


def _board_with_ojama_only() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    grid[BOARD_ROWS - 1][0] = COLOR_OJAMA
    grid[BOARD_ROWS - 1][1] = COLOR_OJAMA
    return Board.from_list(grid)


def test_count_puyos_empty() -> None:
    assert count_puyos_on_field(_empty_board()) == 0


def test_count_puyos_with_color() -> None:
    assert count_puyos_on_field(_board_with_color()) == 1


def test_count_puyos_excludes_ojama() -> None:
    b = _board_with_ojama_only()
    # おじゃま含めて 2、除外で 0
    assert count_puyos_on_field(b, exclude_ojama=False) == 2
    assert count_puyos_on_field(b, exclude_ojama=True) == 0


def test_is_all_clear_score_zero() -> None:
    """score=0 → 試合冒頭、全消しではない。"""
    r = is_all_clear(_empty_board(), score=0)
    assert not r.is_all_clear
    assert "試合冒頭" in r.reason or r.score == 0


def test_is_all_clear_color_puyo_remaining() -> None:
    """色ぷよが残っている → 全消しではない。"""
    r = is_all_clear(_board_with_color(), score=1000)
    assert not r.is_all_clear
    assert r.n_color_puyo == 1


def test_is_all_clear_after_chain() -> None:
    """色ぷよなし + score>0 → 全消し。"""
    r = is_all_clear(_empty_board(), score=1000)
    assert r.is_all_clear
    assert r.n_color_puyo == 0
    assert r.score == 1000


def test_is_all_clear_with_ojama_only() -> None:
    """おじゃまだけ残っている (色ぷよ 0) → 全消し。"""
    r = is_all_clear(_board_with_ojama_only(), score=500)
    assert r.is_all_clear
    assert r.n_color_puyo == 0
    assert r.n_puyo_on_field == 2  # おじゃま 2 個


def test_is_all_clear_relax_score() -> None:
    """require_score_nonzero=False で score 0 でも判定可能。"""
    r = is_all_clear(
        _empty_board(), score=0,
        require_score_nonzero=False,
    )
    assert r.is_all_clear


def test_result_dataclass() -> None:
    r = AllClearResult(
        is_all_clear=True, score=100, n_puyo_on_field=0,
        n_color_puyo=0, reason="test",
    )
    assert r.is_all_clear
