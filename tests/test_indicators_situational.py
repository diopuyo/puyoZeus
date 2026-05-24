"""Phase H1 戦況・タイミング指標 8 個のテスト (2026-05-08).

時間軸・対戦相互作用を扱う Situational Indicators:
    - SelfChainDurationIndicator (frame 数推定)
    - OppChainDurationIndicator (相手連鎖 frame 数)
    - ChainDurationAdvantageIndicator (応答可能 puyo 数差)
    - HarassEventCount30sIndicator (state-holding stub)
    - EarlyAggressionScoreIndicator (state-holding stub)
    - CounterIgnitionSignalIndicator (state-holding stub)
    - PostAllClearStateIndicator (序盤全消し検出)
    - UpperBoardDensityIndicator (上部 puyo 密度)

state-holding 系 3 指標 (harass_count_30s / early_aggression / counter_ignition)
は API のみ準備し、現状は中立値 0.5 を返す。
"""
from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_RED,
    Board,
)
from src.chain import ChainSimulator
from src.indicators import (
    ChainDurationAdvantageIndicator,
    CounterIgnitionSignalIndicator,
    EarlyAggressionScoreIndicator,
    HarassEventCount30sIndicator,
    INDICATOR_CHAIN_DURATION_ADV,
    INDICATOR_COUNTER_IGNITION,
    INDICATOR_EARLY_AGGRESSION,
    INDICATOR_HARASS_COUNT_30S,
    INDICATOR_OPP_CHAIN_DURATION,
    INDICATOR_POST_ALL_CLEAR,
    INDICATOR_SELF_CHAIN_DURATION,
    INDICATOR_UPPER_DENSITY,
    IndicatorCalculator,
    OppChainDurationIndicator,
    PostAllClearStateIndicator,
    SITUATIONAL_NEUTRAL_SCORE,
    SelfChainDurationIndicator,
    UpperBoardDensityIndicator,
)


# ============================
# fixtures
# ============================


def _empty_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(grid)


def _four_red_row() -> Board:
    """4 連結 1 連鎖発火可能盤面."""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for c in range(4):
        grid[BOARD_ROWS - 1][c] = COLOR_RED
    return Board.from_list(grid)


def _two_chain_board() -> Board:
    """確実に 2 連鎖発火する盤面 (ランダム探索で発見、行 10..12)。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    grid[10] = [COLOR_BLUE, COLOR_BLUE, COLOR_RED, COLOR_BLUE,
                COLOR_EMPTY, COLOR_BLUE]
    grid[11] = [COLOR_BLUE, COLOR_BLUE, COLOR_BLUE, COLOR_BLUE,
                COLOR_EMPTY, COLOR_RED]
    grid[12] = [COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_RED,
                COLOR_RED, COLOR_EMPTY]
    return Board.from_list(grid)


def _upper_filled_board() -> Board:
    """上部 (row 0..3) を puyo で埋めた盤面 (upper_density 高)."""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for r in range(4):
        for c in range(BOARD_COLS):
            grid[r][c] = COLOR_RED
    return Board.from_list(grid)


# ============================
# SelfChainDurationIndicator
# ============================


def test_self_chain_duration_empty_zero() -> None:
    """空盤面は連鎖 0 → frames 0 → score 0."""
    ind = SelfChainDurationIndicator()
    res = ind.compute(_empty_board())
    assert res.name == INDICATOR_SELF_CHAIN_DURATION
    assert res.score == 0.0
    assert res.raw_value == 0.0


def test_self_chain_duration_one_chain_positive() -> None:
    """1 連鎖盤面は frames > 0."""
    ind = SelfChainDurationIndicator()
    sim = ChainSimulator()
    res = ind.compute(_four_red_row(), simulator=sim)
    assert res.raw_value > 0.0
    assert res.score > 0.0


def test_self_chain_duration_two_chain_higher() -> None:
    """2 連鎖は 1 連鎖より frames が大きい."""
    ind = SelfChainDurationIndicator()
    sim = ChainSimulator()
    r1 = ind.compute(_four_red_row(), simulator=sim)
    r2 = ind.compute(_two_chain_board(), simulator=sim)
    assert r2.score > r1.score


def test_self_chain_duration_score_in_range() -> None:
    """0..1 範囲内."""
    ind = SelfChainDurationIndicator()
    sim = ChainSimulator()
    for b in [_empty_board(), _four_red_row(), _two_chain_board()]:
        res = ind.compute(b, simulator=sim)
        assert 0.0 <= res.score <= 1.0


# ============================
# OppChainDurationIndicator
# ============================


def test_opp_chain_duration_no_opponent_zero() -> None:
    """opponent_board=None なら 0."""
    ind = OppChainDurationIndicator()
    res = ind.compute(_empty_board(), opponent_board=None)
    assert res.name == INDICATOR_OPP_CHAIN_DURATION
    assert res.score == 0.0
    assert res.detail["reason"] == "no_opponent_board"


def test_opp_chain_duration_with_opponent_positive() -> None:
    """opponent_board に発火可能盤面を渡すと frames > 0."""
    ind = OppChainDurationIndicator()
    sim = ChainSimulator()
    res = ind.compute(_empty_board(), opponent_board=_four_red_row(),
                      simulator=sim)
    assert res.raw_value > 0.0
    assert res.score > 0.0


def test_opp_chain_duration_score_in_range() -> None:
    """0..1 範囲内."""
    ind = OppChainDurationIndicator()
    sim = ChainSimulator()
    for opp in [_empty_board(), _four_red_row(), _two_chain_board()]:
        res = ind.compute(_empty_board(), opponent_board=opp, simulator=sim)
        assert 0.0 <= res.score <= 1.0


# ============================
# ChainDurationAdvantageIndicator
# ============================


def test_chain_duration_adv_no_opponent_neutral() -> None:
    """opponent_board=None なら中立 0.5."""
    ind = ChainDurationAdvantageIndicator()
    res = ind.compute(_empty_board(), opponent_board=None)
    assert res.name == INDICATOR_CHAIN_DURATION_ADV
    assert res.score == SITUATIONAL_NEUTRAL_SCORE


def test_chain_duration_adv_opponent_chain_high_for_self() -> None:
    """相手が長い連鎖、自分が空 → 自分有利 (応答時間多い) → score > 0.5."""
    ind = ChainDurationAdvantageIndicator()
    sim = ChainSimulator()
    res = ind.compute(_empty_board(), opponent_board=_two_chain_board(),
                      simulator=sim)
    assert res.score > SITUATIONAL_NEUTRAL_SCORE


def test_chain_duration_adv_self_chain_high_disadvantage() -> None:
    """自分が長い連鎖、相手が空 → 自分不利 → score < 0.5."""
    ind = ChainDurationAdvantageIndicator()
    sim = ChainSimulator()
    res = ind.compute(_two_chain_board(), opponent_board=_empty_board(),
                      simulator=sim)
    assert res.score < SITUATIONAL_NEUTRAL_SCORE


def test_chain_duration_adv_score_in_range() -> None:
    """0..1 範囲内."""
    ind = ChainDurationAdvantageIndicator()
    sim = ChainSimulator()
    boards = [_empty_board(), _four_red_row(), _two_chain_board()]
    for self_b in boards:
        for opp_b in boards:
            res = ind.compute(self_b, opponent_board=opp_b, simulator=sim)
            assert 0.0 <= res.score <= 1.0


# ============================
# HarassEventCount30sIndicator (state-holding stub)
# ============================


def test_harass_count_30s_no_param_neutral() -> None:
    """harass_count=None なら中立値 0.5 (state-holding stub)."""
    ind = HarassEventCount30sIndicator()
    res = ind.compute(_empty_board(), harass_count=None)
    assert res.name == INDICATOR_HARASS_COUNT_30S
    assert res.score == SITUATIONAL_NEUTRAL_SCORE
    assert res.detail["stateful"] is False


def test_harass_count_30s_with_param_normalized() -> None:
    """harass_count=2 なら 0..1 に正規化."""
    ind = HarassEventCount30sIndicator()
    res = ind.compute(_empty_board(), harass_count=2)
    assert 0.0 <= res.score <= 1.0
    assert res.detail["stateful"] is True
    assert res.detail["harass_count"] == 2


def test_harass_count_30s_high_count_clamped() -> None:
    """大きい値 (10) は 1.0 にクランプ."""
    ind = HarassEventCount30sIndicator()
    res = ind.compute(_empty_board(), harass_count=10)
    assert res.score == 1.0


# ============================
# EarlyAggressionScoreIndicator (state-holding stub)
# ============================


def test_early_aggression_no_param_neutral() -> None:
    """early_aggression=None なら中立値 0.5."""
    ind = EarlyAggressionScoreIndicator()
    res = ind.compute(_empty_board(), early_aggression=None)
    assert res.name == INDICATOR_EARLY_AGGRESSION
    assert res.score == SITUATIONAL_NEUTRAL_SCORE


def test_early_aggression_with_param_clamped() -> None:
    """early_aggression=0.7 なら 0.7."""
    ind = EarlyAggressionScoreIndicator()
    res = ind.compute(_empty_board(), early_aggression=0.7)
    assert res.score == 0.7
    assert res.detail["stateful"] is True


def test_early_aggression_out_of_range_clamped() -> None:
    """範囲外の値 (1.5) は 1.0 にクランプ."""
    ind = EarlyAggressionScoreIndicator()
    res = ind.compute(_empty_board(), early_aggression=1.5)
    assert res.score == 1.0


# ============================
# CounterIgnitionSignalIndicator (state-holding stub)
# ============================


def test_counter_ignition_no_param_neutral() -> None:
    """counter_signal=None なら中立値 0.5."""
    ind = CounterIgnitionSignalIndicator()
    res = ind.compute(_empty_board(), counter_signal=None)
    assert res.name == INDICATOR_COUNTER_IGNITION
    assert res.score == SITUATIONAL_NEUTRAL_SCORE


def test_counter_ignition_with_param() -> None:
    """counter_signal=0.8 なら 0.8."""
    ind = CounterIgnitionSignalIndicator()
    res = ind.compute(_empty_board(), counter_signal=0.8)
    assert res.score == 0.8
    assert res.detail["stateful"] is True


# ============================
# PostAllClearStateIndicator
# ============================


def test_post_all_clear_no_chain_zero() -> None:
    """発火しない盤面は全消しなし → 0."""
    ind = PostAllClearStateIndicator()
    res = ind.compute(_empty_board())
    assert res.name == INDICATOR_POST_ALL_CLEAR
    assert res.score == 0.0


def test_post_all_clear_one_chain_clears_board() -> None:
    """4 連結だけ盤面: 発火後盤面に残ぷよなし → 全消し検出."""
    ind = PostAllClearStateIndicator()
    sim = ChainSimulator()
    # elapsed_sec=10 (序盤) で全消し → score = 1.0
    res = ind.compute(_four_red_row(), simulator=sim, elapsed_sec=10.0)
    assert res.detail["is_all_clear"] is True
    # 序盤判定で 1.0
    assert res.score == 1.0


def test_post_all_clear_late_game_lower_score() -> None:
    """全消ししたが late game (elapsed > 60s) なら score 0.5."""
    ind = PostAllClearStateIndicator()
    sim = ChainSimulator()
    res = ind.compute(_four_red_row(), simulator=sim, elapsed_sec=120.0)
    assert res.detail["is_all_clear"] is True
    assert res.score == 0.5


def test_post_all_clear_no_param_score_in_range() -> None:
    """elapsed_sec=None でも score 範囲内."""
    ind = PostAllClearStateIndicator()
    sim = ChainSimulator()
    res = ind.compute(_four_red_row(), simulator=sim, elapsed_sec=None)
    assert 0.0 <= res.score <= 1.0


# ============================
# UpperBoardDensityIndicator
# ============================


def test_upper_density_empty_zero() -> None:
    """空盤面は上部密度 0."""
    ind = UpperBoardDensityIndicator()
    res = ind.compute(_empty_board())
    assert res.name == INDICATOR_UPPER_DENSITY
    assert res.score == 0.0


def test_upper_density_bottom_only_zero() -> None:
    """下層だけに puyo がある盤面は上部密度 0."""
    ind = UpperBoardDensityIndicator()
    res = ind.compute(_four_red_row())
    assert res.score == 0.0


def test_upper_density_upper_filled_max() -> None:
    """上 4 段全埋め → score 1.0."""
    ind = UpperBoardDensityIndicator()
    res = ind.compute(_upper_filled_board())
    assert res.score == 1.0


def test_upper_density_score_in_range() -> None:
    """0..1 範囲内."""
    ind = UpperBoardDensityIndicator()
    for b in [_empty_board(), _four_red_row(), _upper_filled_board()]:
        res = ind.compute(b)
        assert 0.0 <= res.score <= 1.0


# ============================
# IndicatorCalculator 統合
# ============================


def test_calc_compute_all_includes_situational_indicators() -> None:
    """compute_all の結果に H1 戦況 8 指標が含まれる."""
    calc = IndicatorCalculator()
    res = calc.compute_all(_empty_board())
    situational = [
        INDICATOR_SELF_CHAIN_DURATION,
        INDICATOR_OPP_CHAIN_DURATION,
        INDICATOR_CHAIN_DURATION_ADV,
        INDICATOR_HARASS_COUNT_30S,
        INDICATOR_EARLY_AGGRESSION,
        INDICATOR_COUNTER_IGNITION,
        INDICATOR_POST_ALL_CLEAR,
        INDICATOR_UPPER_DENSITY,
    ]
    for name in situational:
        assert name in res.results
    # IndicatorSet field 反映
    assert res.self_chain_duration_frames == \
        res.results[INDICATOR_SELF_CHAIN_DURATION].score
    assert res.upper_board_density == \
        res.results[INDICATOR_UPPER_DENSITY].score


def test_calc_compute_all_with_stateful_params() -> None:
    """stateful 注入引数が IndicatorSet に反映される."""
    calc = IndicatorCalculator()
    res = calc.compute_all(
        _empty_board(),
        elapsed_sec=10.0,
        harass_count=3,
        early_aggression=0.7,
        counter_signal=0.4,
    )
    # state-holding 系が中立値ではなくなる
    assert res.harass_event_count_30s != SITUATIONAL_NEUTRAL_SCORE
    assert res.early_aggression_score == 0.7
    assert res.counter_ignition_signal == 0.4


def test_calc_compute_all_default_state_neutral() -> None:
    """state-holding 引数なしなら中立値."""
    calc = IndicatorCalculator()
    res = calc.compute_all(_empty_board())
    assert res.harass_event_count_30s == SITUATIONAL_NEUTRAL_SCORE
    assert res.early_aggression_score == SITUATIONAL_NEUTRAL_SCORE
    assert res.counter_ignition_signal == SITUATIONAL_NEUTRAL_SCORE
