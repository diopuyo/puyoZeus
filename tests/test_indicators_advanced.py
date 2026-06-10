"""
高度戦略指標 (key_flexibility / sub_chain_independence / chain_timing_pressure)
のユニットテスト。

各指標について:
    - レンジ確認 (0〜1)
    - 空盤面で 0 or 中立値
    - 期待される盤面での値の高低
    - エッジケース (窒息盤面・全色未配置など)

ヘルパー関数 _try_drop_one / _strip_main_chain_groups / _min_puyos_to_ignite
についても代表的な振る舞いを確認する。
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
    COLOR_YELLOW,
    Board,
)
from src.chain import ChainSimulator
from src.old.indicators import (
    INDICATOR_CHAIN_TIMING,
    INDICATOR_KEY_FLEXIBILITY,
    INDICATOR_SUB_CHAIN_INDEP,
    ChainTimingPressureIndicator,
    IndicatorCalculator,
    KeyFlexibilityIndicator,
    SubChainIndependenceIndicator,
    _min_puyos_to_ignite,
    _strip_main_chain_groups,
    _try_drop_one,
)


# ============================
# テスト用ヘルパー
# ============================


def empty_grid() -> list[list[int]]:
    """13×6 全空の盤面グリッド。"""
    return [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]


def make_empty_board() -> Board:
    return Board.from_list(empty_grid())


def make_three_red_row_board() -> Board:
    """row=12 に赤3個のみ (4個目で発火する典型的キーぷよ盤面)。"""
    grid = empty_grid()
    grid[12][0] = COLOR_RED
    grid[12][1] = COLOR_RED
    grid[12][2] = COLOR_RED
    return Board.from_list(grid)


def make_full_red_4_board() -> Board:
    """row=12 に赤4個。発火済み or すぐ消える盤面。"""
    grid = empty_grid()
    grid[12][0] = COLOR_RED
    grid[12][1] = COLOR_RED
    grid[12][2] = COLOR_RED
    grid[12][3] = COLOR_RED
    return Board.from_list(grid)


def make_choking_board() -> Board:
    """3列目最上段に puyo を置いて窒息状態にする。"""
    grid = empty_grid()
    grid[0][2] = COLOR_RED
    return Board.from_list(grid)


def make_main_plus_isolated_sub_board() -> Board:
    """
    本線候補 (赤4連結) + 独立した小副砲候補 (青3連結+1ずれ) の盤面。

    撤去すると赤4連結が消えて、青の3連結が残る。
    """
    grid = empty_grid()
    # 本線: 列 0-3 の row=12 に赤 4 連結
    grid[12][0] = COLOR_RED
    grid[12][1] = COLOR_RED
    grid[12][2] = COLOR_RED
    grid[12][3] = COLOR_RED
    # 副砲候補: 列 4,5 に青 (孤立、サイズ < 4 で残る)
    grid[12][4] = COLOR_BLUE
    grid[12][5] = COLOR_BLUE
    grid[11][5] = COLOR_BLUE
    return Board.from_list(grid)


# ============================
# _try_drop_one のテスト
# ============================


def test_try_drop_one_creates_new_board() -> None:
    """元盤面を変更せず新盤面を返す。"""
    b = make_empty_board()
    new_b = _try_drop_one(b, 0, COLOR_RED)
    assert new_b is not None
    # 元盤面は変更されない
    assert b.get(BOARD_ROWS - 1, 0) == COLOR_EMPTY
    # 新盤面の最下段に赤が落下している
    assert new_b.get(BOARD_ROWS - 1, 0) == COLOR_RED


def test_try_drop_one_returns_none_when_full() -> None:
    """満杯列に落とすと None を返す。"""
    grid = empty_grid()
    for r in range(BOARD_ROWS):
        grid[r][0] = COLOR_RED
    b = Board.from_list(grid)
    assert _try_drop_one(b, 0, COLOR_BLUE) is None


# ============================
# _strip_main_chain_groups のテスト
# ============================


def test_strip_main_chain_groups_removes_4plus() -> None:
    """サイズ >= 4 のグループを撤去し、残骸を返す。"""
    sim = ChainSimulator()
    b = make_main_plus_isolated_sub_board()
    residual = _strip_main_chain_groups(b, sim)
    # 赤4連結は消えている
    red_count = sum(
        1
        for r in range(BOARD_ROWS)
        for c in range(BOARD_COLS)
        if residual.get(r, c) == COLOR_RED
    )
    assert red_count == 0
    # 青3連結は残っている
    blue_count = sum(
        1
        for r in range(BOARD_ROWS)
        for c in range(BOARD_COLS)
        if residual.get(r, c) == COLOR_BLUE
    )
    assert blue_count == 3


# ============================
# _min_puyos_to_ignite のテスト
# ============================


def test_min_puyos_to_ignite_returns_one_for_three_row() -> None:
    """赤3個並びは 1 個追加で発火する。"""
    sim = ChainSimulator()
    b = make_three_red_row_board()
    n = _min_puyos_to_ignite(b, sim, base_chain=0, trial_limit=4)
    assert n == 1


def test_min_puyos_to_ignite_returns_max_plus_one_for_empty() -> None:
    """空盤面では 1 puyo 追加でも発火しない (試行上限超過)。"""
    sim = ChainSimulator()
    b = make_empty_board()
    n = _min_puyos_to_ignite(b, sim, base_chain=0, trial_limit=2)
    # 1 puyo では4連結作れない、2 puyo でも作れない
    assert n == 3  # trial_limit + 1


# ============================
# KeyFlexibilityIndicator のテスト
# ============================


def test_key_flexibility_range() -> None:
    """値は 0〜1 の範囲に収まる。"""
    ind = KeyFlexibilityIndicator()
    b = make_three_red_row_board()
    res = ind.compute(b)
    assert 0.0 <= res.score <= 1.0


def test_key_flexibility_high_for_three_connected() -> None:
    """4連結直前 (赤3) では key_flexibility が正値になる。"""
    ind = KeyFlexibilityIndicator()
    b = make_three_red_row_board()
    res = ind.compute(b)
    # 赤を col=3 (またはその他隣接) に置けば 4連結発火する placement が存在
    assert res.score > 0.0
    assert res.detail["extension_count"] > 0


def test_key_flexibility_zero_on_empty_board() -> None:
    """空盤面では 1 puyo 追加では発火不可で 0。"""
    ind = KeyFlexibilityIndicator()
    b = make_empty_board()
    res = ind.compute(b)
    assert res.score == 0.0


# ============================
# SubChainIndependenceIndicator のテスト
# ============================


def test_sub_chain_independence_range() -> None:
    """値は 0〜1 の範囲に収まる。"""
    ind = SubChainIndependenceIndicator()
    b = make_main_plus_isolated_sub_board()
    res = ind.compute(b)
    assert 0.0 <= res.score <= 1.0


def test_sub_chain_independence_zero_on_dead_board() -> None:
    """窒息盤面では副砲評価不可で 0。"""
    ind = SubChainIndependenceIndicator()
    b = make_choking_board()
    res = ind.compute(b)
    assert res.score == 0.0
    assert res.detail.get("board_dead") is True


def test_sub_chain_independence_positive_when_sub_remains() -> None:
    """本線撤去後に副砲候補 (3連結+1で発火) が残ると正値。"""
    ind = SubChainIndependenceIndicator()
    b = make_main_plus_isolated_sub_board()
    res = ind.compute(b)
    # 残骸 (青3連結) に 1 個追加すれば 1 連鎖発火 → best_sub_chain >= 1
    assert res.detail["best_sub_chain"] >= 1
    assert res.score > 0.0


# ============================
# ChainTimingPressureIndicator のテスト
# ============================


def test_chain_timing_pressure_range() -> None:
    """値は 0〜1 の範囲に収まる。"""
    ind = ChainTimingPressureIndicator()
    b = make_three_red_row_board()
    res = ind.compute(b)
    assert 0.0 <= res.score <= 1.0


def test_chain_timing_pressure_high_for_near_ignition() -> None:
    """1 puyo で発火可能な盤面は高評価 (>= 0.7)。"""
    ind = ChainTimingPressureIndicator()
    b = make_three_red_row_board()
    res = ind.compute(b)
    assert res.score >= 0.7
    assert res.detail["min_n"] == 1


def test_chain_timing_pressure_low_for_empty_board() -> None:
    """空盤面では発火不可で低評価 (= 1 - (limit+1)/MAX で 0 寄り)。"""
    ind = ChainTimingPressureIndicator()
    b = make_empty_board()
    res = ind.compute(b)
    # min_n = trial_limit + 1 = 7, 7/6 > 1, clamped to 0
    assert res.score == 0.0


def test_chain_timing_pressure_zero_on_dead_board() -> None:
    """窒息盤面では 0。"""
    ind = ChainTimingPressureIndicator()
    b = make_choking_board()
    res = ind.compute(b)
    assert res.score == 0.0


# ============================
# IndicatorCalculator 統合テスト
# ============================


def test_calculator_includes_advanced_indicators() -> None:
    """IndicatorCalculator が 3 つの新指標を全て計算する。"""
    calc = IndicatorCalculator()
    b = make_three_red_row_board()
    ind_set = calc.compute_all(b)
    # results 辞書に追加されている
    assert INDICATOR_KEY_FLEXIBILITY in ind_set.results
    assert INDICATOR_SUB_CHAIN_INDEP in ind_set.results
    assert INDICATOR_CHAIN_TIMING in ind_set.results
    # 同名属性にも反映されている
    assert ind_set.key_flexibility == ind_set.results[INDICATOR_KEY_FLEXIBILITY].score
    assert ind_set.sub_chain_independence == ind_set.results[INDICATOR_SUB_CHAIN_INDEP].score
    assert ind_set.chain_timing_pressure == ind_set.results[INDICATOR_CHAIN_TIMING].score


def test_calculator_advanced_scores_in_range() -> None:
    """様々な盤面で新指標の値が 0〜1 に収まる。"""
    calc = IndicatorCalculator()
    boards = [
        make_empty_board(),
        make_three_red_row_board(),
        make_full_red_4_board(),
        make_main_plus_isolated_sub_board(),
        make_choking_board(),
    ]
    for b in boards:
        ind_set = calc.compute_all(b)
        for name in (
            INDICATOR_KEY_FLEXIBILITY,
            INDICATOR_SUB_CHAIN_INDEP,
            INDICATOR_CHAIN_TIMING,
        ):
            score = ind_set.results[name].score
            assert 0.0 <= score <= 1.0, f"{name} out of range on {b!r}: {score}"
