"""Phase F (B-4 + C-3) 統合テスト.

  - IndicatorCalculator.compute_all に rotation_score 引数を渡すと
    IndicatorSet.rotation_skill に反映される
  - EXTRA_INDICATOR_NAMES に rotation_skill が含まれる
  - backwards compat: opponent_board=None / rotation_score=0.5 で
    従来挙動が破壊されない
"""

from __future__ import annotations

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_RED,
    Board,
)
from src.old.indicators import (
    EXTRA_INDICATOR_NAMES,
    INDICATOR_NEXT_ACCEPTANCE,
    INDICATOR_ROTATION_SKILL,
    IndicatorCalculator,
    IndicatorSet,
    RotationSkillIndicator,
)


def _empty_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(grid)


def _three_red_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for c in range(3):
        grid[BOARD_ROWS - 1][c] = COLOR_RED
    return Board.from_list(grid)


# ============================
# RotationSkillIndicator (degenerate)
# ============================


def test_rotation_skill_indicator_returns_score_as_is() -> None:
    """rotation_score をそのまま score として返す."""
    ind = RotationSkillIndicator()
    res = ind.compute(_empty_board(), rotation_score=0.7)
    assert res.score == 0.7
    assert res.name == INDICATOR_ROTATION_SKILL
    assert res.detail["rotation_score"] == 0.7


def test_rotation_skill_indicator_clamps_above_one() -> None:
    """rotation_score > 1 でも 1.0 にクランプされる."""
    ind = RotationSkillIndicator()
    res = ind.compute(_empty_board(), rotation_score=1.5)
    assert res.score == 1.0
    assert res.raw_value == 1.5


def test_rotation_skill_indicator_clamps_below_zero() -> None:
    """rotation_score < 0 でも 0.0 にクランプされる."""
    ind = RotationSkillIndicator()
    res = ind.compute(_empty_board(), rotation_score=-0.3)
    assert res.score == 0.0


def test_rotation_skill_indicator_default_neutral() -> None:
    """rotation_score 未指定時はデフォルト 0.5 (neutral)."""
    ind = RotationSkillIndicator()
    res = ind.compute(_empty_board())
    assert res.score == 0.5


# ============================
# EXTRA_INDICATOR_NAMES
# ============================


def test_rotation_skill_in_extra_indicator_names() -> None:
    """EXTRA_INDICATOR_NAMES に rotation_skill が含まれる."""
    assert INDICATOR_ROTATION_SKILL in EXTRA_INDICATOR_NAMES


# ============================
# IndicatorCalculator.compute_all
# ============================


def test_compute_all_passes_rotation_score_through() -> None:
    """rotation_score 引数が IndicatorSet.rotation_skill に反映される."""
    calc = IndicatorCalculator()
    res: IndicatorSet = calc.compute_all(_empty_board(), rotation_score=0.8)
    assert res.rotation_skill == 0.8
    # results[INDICATOR_ROTATION_SKILL] とも一致
    assert res.results[INDICATOR_ROTATION_SKILL].score == 0.8


def test_compute_all_default_rotation_skill_is_neutral() -> None:
    """rotation_score 未指定なら IndicatorSet.rotation_skill = 0.5."""
    calc = IndicatorCalculator()
    res = calc.compute_all(_empty_board())
    assert res.rotation_skill == 0.5


def test_compute_all_returns_all_extras_including_rotation() -> None:
    """compute_all の results dict に rotation_skill キーが含まれる.

    next_acceptance は IndicatorSet.next_acceptance 属性で別管理のため
    results dict には入らない (既存設計、IndicatorSet.next_acceptance を使う)。
    """
    calc = IndicatorCalculator()
    res = calc.compute_all(_three_red_board())
    for name in EXTRA_INDICATOR_NAMES:
        if name == INDICATOR_NEXT_ACCEPTANCE:
            continue  # 別 attr 管理 (results dict に含めない設計)
        assert name in res.results, f"{name} not in results"
    # rotation_skill は results にも含まれる
    assert INDICATOR_ROTATION_SKILL in res.results


# ============================
# backwards compat
# ============================


def test_compute_all_backwards_compat_no_opponent_no_rotation() -> None:
    """opponent_board / rotation_score を渡さなくても従来通り動作する."""
    calc = IndicatorCalculator()
    # 旧シグネチャ呼び出し (positional / keyword)
    res = calc.compute_all(_three_red_board())
    # 全 EXTRA + 主指標が dict に揃う (next_acceptance のみ別 attr)
    for name in EXTRA_INDICATOR_NAMES:
        if name == INDICATOR_NEXT_ACCEPTANCE:
            continue
        assert name in res.results
    # 主指標も健在
    assert "main_chain_maturity" in res.results
    assert "extension_potential" in res.results
    assert "second_chain_potential" in res.results


def test_compute_all_score_in_unit_range_with_rotation() -> None:
    """rotation_score 指定時も全指標 score が [0, 1] 範囲."""
    calc = IndicatorCalculator()
    res = calc.compute_all(_three_red_board(), rotation_score=0.3)
    for name, result in res.results.items():
        assert 0.0 <= result.score <= 1.0, f"{name} out of range: {result.score}"
