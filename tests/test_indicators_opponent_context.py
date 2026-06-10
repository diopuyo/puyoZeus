"""Phase F (C-3) 指標の opponent_board コンテキスト拡張テスト.

対象:
  - ChainTimingPressureIndicator (相対 pressure)
  - SecondChainPotentialIndicator (相手脅威時 boost)
  - ExtensionPotentialIndicator (相手高 chain_maturity 時 decay)

backwards compat: opponent_board=None で従来挙動を維持する。
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
from src.chain import ChainSimulator
from src.old.indicators import (
    INDICATOR_CHAIN_TIMING,
    INDICATOR_EXTENSION,
    INDICATOR_SECOND,
    ChainTimingPressureIndicator,
    ExtensionPotentialIndicator,
    SecondChainPotentialIndicator,
)


# ============================
# テスト用盤面ヘルパー
# ============================


def _empty_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(grid)


def _three_red_board() -> Board:
    """3 連結のみ。1 puyo 追加で 4 連結発火 (min_n = 1)。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    grid[BOARD_ROWS - 1][0] = COLOR_RED
    grid[BOARD_ROWS - 1][1] = COLOR_RED
    grid[BOARD_ROWS - 1][2] = COLOR_RED
    return Board.from_list(grid)


def _far_from_fire_board() -> Board:
    """発火に複数 puyo 必要 (min_n 大)。1 puyo の単独配置。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    grid[BOARD_ROWS - 1][0] = COLOR_RED
    return Board.from_list(grid)


def _high_chain_board() -> Board:
    """発火可能な 4 連鎖クラス盤面 (相手の脅威源)。

    階段 4 連鎖のシンプル構造。
    """
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    # col 0 (rows 5-12): Y Y B B G G R R
    seq = [
        COLOR_YELLOW, COLOR_YELLOW,
        COLOR_BLUE, COLOR_BLUE,
        COLOR_GREEN, COLOR_GREEN,
        COLOR_RED, COLOR_RED,
    ]
    for i, c in enumerate(seq):
        grid[BOARD_ROWS - 1 - i][0] = c
    # col 1-3 (rows 9-12): Y B G R
    for offset, color in enumerate(
        [COLOR_YELLOW, COLOR_BLUE, COLOR_GREEN, COLOR_RED]
    ):
        for col in range(1, 4):
            grid[BOARD_ROWS - 1 - offset][col] = color
    return Board.from_list(grid)


# ============================
# ChainTimingPressureIndicator
# ============================


def test_chain_timing_no_opponent_board_keeps_legacy_behavior() -> None:
    """opponent_board=None なら従来通りの単独評価."""
    ind = ChainTimingPressureIndicator()
    sim = ChainSimulator()
    res_with = ind.compute(_three_red_board(), simulator=sim)
    res_without = ind.compute(_three_red_board(), simulator=sim,
                              opponent_board=None)
    assert res_with.score == res_without.score
    assert res_with.name == INDICATOR_CHAIN_TIMING
    assert res_with.detail.get("opp_min_n") is None
    assert res_with.detail.get("relative_mode", False) is False


def test_chain_timing_self_faster_than_opponent_score_higher_than_neutral() -> None:
    """自分が相手より早く発火可能なら relative pressure が 0.5 以上."""
    ind = ChainTimingPressureIndicator()
    sim = ChainSimulator()
    res = ind.compute(
        _three_red_board(), simulator=sim,
        opponent_board=_far_from_fire_board(),
    )
    assert res.detail.get("relative_mode") is True
    assert res.detail.get("opp_min_n") is not None
    # 自分 (min_n=1) < 相手 (min_n>=2) → 相対 pressure > 0.5
    assert res.score > 0.5


def test_chain_timing_self_slower_than_opponent_score_lower() -> None:
    """自分が相手より遅い場合は relative pressure が 0.5 以下."""
    ind = ChainTimingPressureIndicator()
    sim = ChainSimulator()
    res = ind.compute(
        _far_from_fire_board(), simulator=sim,
        opponent_board=_three_red_board(),
    )
    assert res.score < 0.5


def test_chain_timing_score_within_unit_range() -> None:
    """relative pressure は 0〜1 にクランプされる."""
    ind = ChainTimingPressureIndicator()
    sim = ChainSimulator()
    res = ind.compute(
        _empty_board(), simulator=sim,
        opponent_board=_three_red_board(),
    )
    assert 0.0 <= res.score <= 1.0


# ============================
# SecondChainPotentialIndicator
# ============================


def test_second_chain_no_opponent_keeps_legacy() -> None:
    """opponent_board=None なら旧挙動 (boost 無し)."""
    ind = SecondChainPotentialIndicator()
    sim = ChainSimulator()
    res_legacy = ind.compute(_three_red_board(), simulator=sim)
    res_none = ind.compute(_three_red_board(), simulator=sim,
                           opponent_board=None)
    assert res_legacy.score == res_none.score
    assert res_legacy.name == INDICATOR_SECOND


def test_second_chain_high_threat_opponent_boosts_score() -> None:
    """相手が高連鎖を抱えると score が boost される."""
    ind = SecondChainPotentialIndicator()
    sim = ChainSimulator()
    base = ind.compute(_three_red_board(), simulator=sim).score
    boosted = ind.compute(
        _three_red_board(), simulator=sim,
        opponent_board=_high_chain_board(),
    ).score
    # boost が走った場合 score が増えるか、すでに 1.0 で頭打ちか
    assert boosted >= base
    # 1.0 を超えないこと
    assert boosted <= 1.0


def test_second_chain_low_threat_opponent_no_boost() -> None:
    """相手が脅威 (連鎖) を持たないなら boost 無しで base と一致."""
    ind = SecondChainPotentialIndicator()
    sim = ChainSimulator()
    base = ind.compute(_three_red_board(), simulator=sim).score
    no_boost = ind.compute(
        _three_red_board(), simulator=sim,
        opponent_board=_far_from_fire_board(),
    ).score
    assert no_boost == base


# ============================
# ExtensionPotentialIndicator
# ============================


def test_extension_no_opponent_keeps_legacy() -> None:
    """opponent_board=None で減衰なし (decay=1.0)、score 不変."""
    ind = ExtensionPotentialIndicator()
    sim = ChainSimulator()
    res_legacy = ind.compute(_three_red_board(), simulator=sim)
    res_none = ind.compute(_three_red_board(), simulator=sim,
                           opponent_board=None)
    assert res_legacy.score == res_none.score
    assert res_legacy.name == INDICATOR_EXTENSION


def test_extension_high_chain_opponent_decays_score() -> None:
    """相手が高 chain_maturity の場合 score が減衰する (or 不変)."""
    ind = ExtensionPotentialIndicator()
    sim = ChainSimulator()
    base = ind.compute(_three_red_board(), simulator=sim).score
    decayed = ind.compute(
        _three_red_board(), simulator=sim,
        opponent_board=_high_chain_board(),
    ).score
    # decay により improvement_ratio が縮小 → score 同等以下
    assert decayed <= base


def test_extension_decay_factor_in_detail() -> None:
    """detail に opp_decay と opp_chain_count が含まれる."""
    ind = ExtensionPotentialIndicator()
    sim = ChainSimulator()
    res = ind.compute(
        _three_red_board(), simulator=sim,
        opponent_board=_high_chain_board(),
    )
    assert "opp_decay" in res.detail
    assert "opp_chain_count" in res.detail
    # decay は [0.7, 1.0] の範囲 (0.3 が最大減衰)
    assert 0.7 <= res.detail["opp_decay"] <= 1.0
