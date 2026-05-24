"""Phase H1 形分類指標 (gtr_orientation) のテスト (2026-05-08).

GtrOrientationIndicator: form_gtr テンプレ評価結果から
先折り (col 0-1) / 後折り (col 4-5) / 自由形 を判定する。
"""
from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_RED,
    Board,
)
from src.form_templates import GTR_TEMPLATE
from src.indicators import (
    GTR_ORIENTATION_BACK,
    GTR_ORIENTATION_FREE,
    GTR_ORIENTATION_FRONT,
    GTR_ORIENTATION_SCORE_BACK,
    GTR_ORIENTATION_SCORE_FREE,
    GTR_ORIENTATION_SCORE_FRONT,
    GtrOrientationIndicator,
    INDICATOR_GTR_ORIENTATION,
    IndicatorCalculator,
)


# ============================
# fixtures
# ============================


def _empty_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(grid)


def _build_gtr_template_board(mirror: bool = False) -> Board:
    """GTR テンプレ通りの盤面を構築 (色等価クラス A=Red, B=Blue, C=Green).

    mirror=True で _mirror_template_cells と一致するレイアウト。
    """
    from src.form_templates import _mirror_template_cells
    color_map = {"A": COLOR_RED, "B": COLOR_BLUE, "C": COLOR_GREEN}
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    cells = GTR_TEMPLATE.cells
    if mirror:
        cells = _mirror_template_cells(cells)
    for ((r_from_bottom, c), cls) in cells:
        actual_row = BOARD_ROWS - 1 - r_from_bottom
        grid[actual_row][c] = color_map[cls]
    return Board.from_list(grid)


# ============================
# GtrOrientationIndicator
# ============================


def test_gtr_orientation_empty_free() -> None:
    """空盤面は GTR 不一致 → free (score 0.0, code 2)."""
    ind = GtrOrientationIndicator()
    res = ind.compute(_empty_board())
    assert res.name == INDICATOR_GTR_ORIENTATION
    assert res.score == GTR_ORIENTATION_SCORE_FREE
    assert res.detail["orientation_code"] == GTR_ORIENTATION_FREE


def test_gtr_orientation_front_template_high() -> None:
    """1P 側 GTR テンプレ盤面は先折り判定."""
    ind = GtrOrientationIndicator()
    board = _build_gtr_template_board(mirror=False)
    res = ind.compute(board)
    # GTR テンプレが col 0..3 (左下) で組まれている → 先折り
    assert res.detail["orientation_code"] == GTR_ORIENTATION_FRONT
    assert res.score == GTR_ORIENTATION_SCORE_FRONT
    assert res.detail["orientation_label"] == "front"


def test_gtr_orientation_back_template_mirror() -> None:
    """2P 側 (mirror) GTR テンプレ盤面は後折り判定."""
    ind = GtrOrientationIndicator()
    board = _build_gtr_template_board(mirror=True)
    res = ind.compute(board)
    # mirror 配置 → mirror_used=True で後折り
    assert res.detail["orientation_code"] == GTR_ORIENTATION_BACK
    assert res.score == GTR_ORIENTATION_SCORE_BACK
    assert res.detail["orientation_label"] == "back"
    assert res.detail["mirror_used"] is True


def test_gtr_orientation_score_in_range() -> None:
    """全テスト盤面で 0..1 範囲内."""
    ind = GtrOrientationIndicator()
    boards = [
        _empty_board(),
        _build_gtr_template_board(mirror=False),
        _build_gtr_template_board(mirror=True),
    ]
    for b in boards:
        res = ind.compute(b)
        assert 0.0 <= res.score <= 1.0


def test_calc_compute_all_includes_gtr_orientation() -> None:
    """compute_all の結果に gtr_orientation が含まれる."""
    calc = IndicatorCalculator()
    res = calc.compute_all(_empty_board())
    assert INDICATOR_GTR_ORIENTATION in res.results
    # IndicatorSet field 反映
    assert res.gtr_orientation == res.results[INDICATOR_GTR_ORIENTATION].score
