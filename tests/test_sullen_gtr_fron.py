"""B-1.b 追加形 (Sullen GTR / Fron) のテスト (2026-05-09 追加).

citrus610/ama (Puyo Puyo Tsu AI) form.h 由来の SGTR / FRON 派生形を
本プロジェクト互換の簡略テンプレで実装。GTR と区別される配置で
完全一致時に score=1.0、空盤面で 0.0、1P/2P mirror で対称、
等価色 permutation で同スコアを保つことを検証する。
"""
from __future__ import annotations

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    Board,
)
from src.form_templates import (
    FRON_TEMPLATE,
    SULLEN_GTR_TEMPLATE,
    best_template_score,
    template_score,
)
from src.indicators import (
    INDICATOR_FORM_FRON,
    INDICATOR_FORM_SULLEN_GTR,
    EXTRA_INDICATOR_NAMES,
    FORM_TEMPLATE_INDICATOR_NAMES,
    FronCompletenessIndicator,
    IndicatorCalculator,
    SullenGtrCompletenessIndicator,
)


# ============================
# helper
# ============================


def _empty_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(grid)


def _set_bot(grid: list[list[int]], row_from_bottom: int, col: int, color: int) -> None:
    """下から row_from_bottom の絶対位置にセル設定."""
    grid[BOARD_ROWS - 1 - row_from_bottom][col] = color


def _sullen_gtr_board(
    a: int = COLOR_RED, b: int = COLOR_BLUE, c: int = COLOR_GREEN,
) -> Board:
    """Sullen GTR テンプレ通り左下に配置した盤面.

    cells (row_from_bottom, col, class):
        (0,0)A (1,0)A (2,0)B
        (0,1)A (1,1)B (2,1)B
        (0,2)B (1,2)C
        (0,3)C
    """
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    _set_bot(grid, 0, 0, a)
    _set_bot(grid, 1, 0, a)
    _set_bot(grid, 2, 0, b)
    _set_bot(grid, 0, 1, a)
    _set_bot(grid, 1, 1, b)
    _set_bot(grid, 2, 1, b)
    _set_bot(grid, 0, 2, b)
    _set_bot(grid, 1, 2, c)
    _set_bot(grid, 0, 3, c)
    return Board.from_list(grid)


def _fron_board(
    a: int = COLOR_RED, b: int = COLOR_BLUE,
    c: int = COLOR_GREEN, d: int = COLOR_YELLOW,
) -> Board:
    """Fron テンプレ通り左下に配置した盤面.

    cells (row_from_bottom, col, class):
        (0,0)A (1,0)A (2,0)B
        (0,1)A (1,1)B (2,1)B
        (0,2)B (1,2)C (2,2)B
        (0,3)C (1,3)D
    """
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    _set_bot(grid, 0, 0, a)
    _set_bot(grid, 1, 0, a)
    _set_bot(grid, 2, 0, b)
    _set_bot(grid, 0, 1, a)
    _set_bot(grid, 1, 1, b)
    _set_bot(grid, 2, 1, b)
    _set_bot(grid, 0, 2, b)
    _set_bot(grid, 1, 2, c)
    _set_bot(grid, 2, 2, b)
    _set_bot(grid, 0, 3, c)
    _set_bot(grid, 1, 3, d)
    return Board.from_list(grid)


# ============================
# template_score: 完全一致 / 空 / mirror / 色 permutation
# ============================


def test_sullen_gtr_perfect_match() -> None:
    """Sullen GTR テンプレ通り → score=1.0."""
    score, detail = template_score(_sullen_gtr_board(), SULLEN_GTR_TEMPLATE)
    assert score == 1.0
    assert detail["matched"] == detail["total"] == 9


def test_fron_perfect_match() -> None:
    """Fron テンプレ通り → score=1.0."""
    score, detail = template_score(_fron_board(), FRON_TEMPLATE)
    assert score == 1.0
    assert detail["matched"] == detail["total"] == 11


def test_sullen_gtr_empty_board_zero() -> None:
    """空盤面では Sullen GTR score=0.0."""
    score, _ = template_score(_empty_board(), SULLEN_GTR_TEMPLATE)
    assert score == 0.0


def test_fron_empty_board_zero() -> None:
    """空盤面では Fron score=0.0."""
    score, _ = template_score(_empty_board(), FRON_TEMPLATE)
    assert score == 0.0


def test_sullen_gtr_color_permutation_invariance() -> None:
    """A/B/C 色の permutation で同じ score を返す (色対称性)."""
    base, _ = template_score(_sullen_gtr_board(), SULLEN_GTR_TEMPLATE)
    perm, _ = template_score(
        _sullen_gtr_board(a=COLOR_BLUE, b=COLOR_GREEN, c=COLOR_RED),
        SULLEN_GTR_TEMPLATE,
    )
    perm2, _ = template_score(
        _sullen_gtr_board(a=COLOR_GREEN, b=COLOR_YELLOW, c=COLOR_BLUE),
        SULLEN_GTR_TEMPLATE,
    )
    assert base == perm == perm2 == 1.0


def test_fron_color_permutation_invariance() -> None:
    """Fron も A/B/C/D 色 permutation で同じ score."""
    base, _ = template_score(_fron_board(), FRON_TEMPLATE)
    perm, _ = template_score(
        _fron_board(
            a=COLOR_YELLOW, b=COLOR_RED, c=COLOR_BLUE, d=COLOR_GREEN,
        ),
        FRON_TEMPLATE,
    )
    assert base == perm == 1.0


def test_sullen_gtr_mirror_2p() -> None:
    """2P 側に水平ミラーで配置 → mirror=True で 1.0."""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    # SGTR mirror cells: max_col=3, col c -> 3-c
    # ((0,3)A (1,3)A (2,3)B) ((0,2)A (1,2)B (2,2)B)
    # ((0,1)B (1,1)C) ((0,0)C)
    _set_bot(grid, 0, 3, COLOR_RED)
    _set_bot(grid, 1, 3, COLOR_RED)
    _set_bot(grid, 2, 3, COLOR_BLUE)
    _set_bot(grid, 0, 2, COLOR_RED)
    _set_bot(grid, 1, 2, COLOR_BLUE)
    _set_bot(grid, 2, 2, COLOR_BLUE)
    _set_bot(grid, 0, 1, COLOR_BLUE)
    _set_bot(grid, 1, 1, COLOR_GREEN)
    _set_bot(grid, 0, 0, COLOR_GREEN)
    board = Board.from_list(grid)
    score_mirror, _ = template_score(board, SULLEN_GTR_TEMPLATE, mirror=True)
    assert score_mirror == 1.0
    score_no, _ = template_score(board, SULLEN_GTR_TEMPLATE, mirror=False)
    assert score_no < 1.0


def test_fron_mirror_2p() -> None:
    """2P 側に Fron をミラー配置 → mirror=True で 1.0."""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    # FRON mirror: col c -> 3-c
    # ((0,3)A (1,3)A (2,3)B) ((0,2)A (1,2)B (2,2)B)
    # ((0,1)B (1,1)C (2,1)B) ((0,0)C (1,0)D)
    _set_bot(grid, 0, 3, COLOR_RED)
    _set_bot(grid, 1, 3, COLOR_RED)
    _set_bot(grid, 2, 3, COLOR_BLUE)
    _set_bot(grid, 0, 2, COLOR_RED)
    _set_bot(grid, 1, 2, COLOR_BLUE)
    _set_bot(grid, 2, 2, COLOR_BLUE)
    _set_bot(grid, 0, 1, COLOR_BLUE)
    _set_bot(grid, 1, 1, COLOR_GREEN)
    _set_bot(grid, 2, 1, COLOR_BLUE)
    _set_bot(grid, 0, 0, COLOR_GREEN)
    _set_bot(grid, 1, 0, COLOR_YELLOW)
    board = Board.from_list(grid)
    score_mirror, _ = template_score(board, FRON_TEMPLATE, mirror=True)
    assert score_mirror == 1.0


def test_sullen_gtr_best_template_returns_one() -> None:
    """best_template_score (1P/2P 自動選択) でも 1.0."""
    score, _ = best_template_score(_sullen_gtr_board(), SULLEN_GTR_TEMPLATE)
    assert score == 1.0


def test_fron_best_template_returns_one() -> None:
    score, _ = best_template_score(_fron_board(), FRON_TEMPLATE)
    assert score == 1.0


# ============================
# Indicator クラス
# ============================


def test_sullen_gtr_indicator_name() -> None:
    ind = SullenGtrCompletenessIndicator()
    assert ind.name == INDICATOR_FORM_SULLEN_GTR


def test_fron_indicator_name() -> None:
    ind = FronCompletenessIndicator()
    assert ind.name == INDICATOR_FORM_FRON


def test_sullen_gtr_indicator_perfect() -> None:
    ind = SullenGtrCompletenessIndicator()
    res = ind.compute(_sullen_gtr_board())
    assert res.score == 1.0
    assert res.detail["template"] == "sullen_gtr"


def test_fron_indicator_perfect() -> None:
    ind = FronCompletenessIndicator()
    res = ind.compute(_fron_board())
    assert res.score == 1.0
    assert res.detail["template"] == "fron"


def test_sullen_gtr_indicator_empty_zero() -> None:
    ind = SullenGtrCompletenessIndicator()
    res = ind.compute(_empty_board())
    assert res.score == 0.0


def test_fron_indicator_empty_zero() -> None:
    ind = FronCompletenessIndicator()
    res = ind.compute(_empty_board())
    assert res.score == 0.0


# ============================
# IndicatorCalculator 統合
# ============================


def test_calc_compute_all_includes_sullen_gtr_and_fron() -> None:
    """compute_all 結果に form_sullen_gtr / form_fron 属性が含まれる."""
    calc = IndicatorCalculator()
    res = calc.compute_all(_sullen_gtr_board())
    # SGTR テンプレ通り → form_sullen_gtr=1.0
    assert res.form_sullen_gtr == 1.0
    # 既存 4 個も互換動作 (regression)
    assert 0.0 <= res.form_gtr <= 1.0
    assert 0.0 <= res.form_llr <= 1.0
    assert 0.0 <= res.form_staircase <= 1.0
    assert 0.0 <= res.form_zabuton <= 1.0
    assert 0.0 <= res.form_fron <= 1.0


def test_calc_results_include_new_form_indicators() -> None:
    """results 辞書に新指標 2 個が含まれる."""
    calc = IndicatorCalculator()
    res = calc.compute_all(_empty_board())
    assert INDICATOR_FORM_SULLEN_GTR in res.results
    assert INDICATOR_FORM_FRON in res.results


def test_extra_indicator_names_contains_new_forms() -> None:
    """EXTRA_INDICATOR_NAMES (順序保持) の末尾近辺に 2 個が含まれる.

    既存 LEARNED_WEIGHTS_* 互換性確認: 新指標は dict キーに含まれなくても OK.
    """
    assert INDICATOR_FORM_SULLEN_GTR in EXTRA_INDICATOR_NAMES
    assert INDICATOR_FORM_FRON in EXTRA_INDICATOR_NAMES


def test_form_template_indicator_names_extends() -> None:
    """FORM_TEMPLATE_INDICATOR_NAMES に 2 個追加 (4 → 6)."""
    assert len(FORM_TEMPLATE_INDICATOR_NAMES) == 6
    assert INDICATOR_FORM_SULLEN_GTR in FORM_TEMPLATE_INDICATOR_NAMES
    assert INDICATOR_FORM_FRON in FORM_TEMPLATE_INDICATOR_NAMES


# ============================
# 既存 LEARNED_WEIGHTS_* 互換性 (regression)
# ============================


def test_existing_learned_weights_unchanged_after_form_addition() -> None:
    """既存 LEARNED_WEIGHTS_GLOBAL に新指標を含まなくても Scorer がエラーにならない.

    backwards compat: dict 形式の既存重みは新指標キーが無いことを前提とし、
    Scorer._extra_diff は self._weights に存在しないキーを skip する。
    """
    from src.scorer import LEARNED_WEIGHTS_GLOBAL, Scorer
    # 新指標は LEARNED_WEIGHTS_GLOBAL に含まれない
    assert INDICATOR_FORM_SULLEN_GTR not in LEARNED_WEIGHTS_GLOBAL
    assert INDICATOR_FORM_FRON not in LEARNED_WEIGHTS_GLOBAL
    # 既存重みで Scorer が問題なく初期化できる (validate_weight_keys が通る)
    scorer = Scorer(weights=LEARNED_WEIGHTS_GLOBAL)
    assert scorer is not None
