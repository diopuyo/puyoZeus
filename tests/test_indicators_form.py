"""I-J 形テンプレ完成度指標 (B-1、2026-05-06) のテスト.

src.indicators の 4 形テンプレ Indicator が IndicatorCalculator から
正しく取り出せ、IndicatorSet の form_* 属性に反映されることを確認。
"""
from __future__ import annotations

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_RED,
    Board,
)
from src.old.indicators import (
    FORM_TEMPLATE_INDICATOR_NAMES,
    INDICATOR_FORM_GTR,
    INDICATOR_FORM_LLR,
    INDICATOR_FORM_STAIRCASE,
    INDICATOR_FORM_ZABUTON,
    GtrCompletenessIndicator,
    IndicatorCalculator,
    LlrCompletenessIndicator,
    StaircaseCompletenessIndicator,
    ZabutonCompletenessIndicator,
)


def _empty_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(grid)


def _gtr_board() -> Board:
    """GTR テンプレ通り左下に配置した盤面."""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    # GTR cells (row_from_bottom, col, class):
    # ((0,0)A, (1,0)A, (2,0)B, (0,1)A, (1,1)B, (2,1)C, (0,2)B, (1,2)C, (0,3)B)
    grid[BOARD_ROWS - 1][0] = COLOR_RED      # (0,0) A
    grid[BOARD_ROWS - 2][0] = COLOR_RED      # (1,0) A
    grid[BOARD_ROWS - 3][0] = COLOR_BLUE     # (2,0) B
    grid[BOARD_ROWS - 1][1] = COLOR_RED      # (0,1) A
    grid[BOARD_ROWS - 2][1] = COLOR_BLUE     # (1,1) B
    grid[BOARD_ROWS - 3][1] = COLOR_GREEN    # (2,1) C
    grid[BOARD_ROWS - 1][2] = COLOR_BLUE     # (0,2) B
    grid[BOARD_ROWS - 2][2] = COLOR_GREEN    # (1,2) C
    grid[BOARD_ROWS - 1][3] = COLOR_BLUE     # (0,3) B
    return Board.from_list(grid)


# ============================
# 単体 Indicator
# ============================


def test_gtr_indicator_name() -> None:
    ind = GtrCompletenessIndicator()
    assert ind.name == INDICATOR_FORM_GTR


def test_gtr_indicator_perfect_match() -> None:
    """GTR テンプレ通り → score=1.0."""
    ind = GtrCompletenessIndicator()
    res = ind.compute(_gtr_board())
    assert res.score == 1.0
    assert res.detail["template"] == "gtr"


def test_gtr_indicator_empty_board_zero() -> None:
    ind = GtrCompletenessIndicator()
    res = ind.compute(_empty_board())
    assert res.score == 0.0


def test_llr_indicator_name() -> None:
    ind = LlrCompletenessIndicator()
    assert ind.name == INDICATOR_FORM_LLR


def test_staircase_indicator_name() -> None:
    ind = StaircaseCompletenessIndicator()
    assert ind.name == INDICATOR_FORM_STAIRCASE


def test_zabuton_indicator_name() -> None:
    ind = ZabutonCompletenessIndicator()
    assert ind.name == INDICATOR_FORM_ZABUTON


def test_all_form_indicators_in_range() -> None:
    """全 4 形テンプレ指標が 0..1 の範囲を返す."""
    indicators = [
        GtrCompletenessIndicator(),
        LlrCompletenessIndicator(),
        StaircaseCompletenessIndicator(),
        ZabutonCompletenessIndicator(),
    ]
    for board in [_empty_board(), _gtr_board()]:
        for ind in indicators:
            res = ind.compute(board)
            assert 0.0 <= res.score <= 1.0


# ============================
# IndicatorCalculator 統合
# ============================


def test_calc_compute_all_form_templates() -> None:
    """IndicatorCalculator.compute_all が form_* 属性を返す."""
    calc = IndicatorCalculator()
    res = calc.compute_all(_gtr_board())
    # GTR テンプレ通りなので form_gtr=1.0
    assert res.form_gtr == 1.0
    # 他のテンプレは 1.0 未満
    assert res.form_llr < 1.0
    assert res.form_staircase < 1.0
    assert res.form_zabuton < 1.0


def test_calc_results_include_form_indicators() -> None:
    """compute_all の results 辞書に 4 形テンプレ指標が含まれる."""
    calc = IndicatorCalculator()
    res = calc.compute_all(_empty_board())
    for name in FORM_TEMPLATE_INDICATOR_NAMES:
        assert name in res.results, f"{name} が results に含まれない"


def test_form_template_indicator_names_count() -> None:
    """FORM_TEMPLATE_INDICATOR_NAMES は 6 個 (B-1.b で SGTR/Fron 追加)."""
    assert len(FORM_TEMPLATE_INDICATOR_NAMES) == 6
    assert INDICATOR_FORM_GTR in FORM_TEMPLATE_INDICATOR_NAMES
    assert INDICATOR_FORM_LLR in FORM_TEMPLATE_INDICATOR_NAMES
    assert INDICATOR_FORM_STAIRCASE in FORM_TEMPLATE_INDICATOR_NAMES
    assert INDICATOR_FORM_ZABUTON in FORM_TEMPLATE_INDICATOR_NAMES
    # B-1.b 追加 (2026-05-09)
    from src.old.indicators import INDICATOR_FORM_FRON, INDICATOR_FORM_SULLEN_GTR
    assert INDICATOR_FORM_SULLEN_GTR in FORM_TEMPLATE_INDICATOR_NAMES
    assert INDICATOR_FORM_FRON in FORM_TEMPLATE_INDICATOR_NAMES


def test_extra_indicator_names_includes_form() -> None:
    """EXTRA_INDICATOR_NAMES に 4 形テンプレ指標が含まれる."""
    from src.old.indicators import EXTRA_INDICATOR_NAMES
    for name in FORM_TEMPLATE_INDICATOR_NAMES:
        assert name in EXTRA_INDICATOR_NAMES, f"{name} が EXTRA に含まれない"
