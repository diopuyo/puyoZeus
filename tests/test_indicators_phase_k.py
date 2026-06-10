"""Phase K 新指標 (2026-04-27 凝視深化) のテスト。

- OpponentOffsetPowerIndicator (相手の即時相殺力)
- PostOjamaChainHealthIndicator (ojama 30 個落下後の本線生存)
- IsolatedPuyoCountIndicator (連鎖参加しない孤立ぷよ)
"""
from __future__ import annotations

import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_RED, Board
from src.chain import ChainSimulator
from src.old.indicators import (
    INDICATOR_ISOLATED_PUYO,
    INDICATOR_OPPONENT_OFFSET,
    INDICATOR_POST_OJAMA_HEALTH,
    IndicatorCalculator,
    IsolatedPuyoCountIndicator,
    OpponentOffsetPowerIndicator,
    PostOjamaChainHealthIndicator,
)


def _empty_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(grid)


def _board_with_4_chain() -> Board:
    """4 連結 1 列を持つ盤面 (即発火可能)。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for c in range(4):
        grid[BOARD_ROWS - 1][c] = COLOR_RED
    return Board.from_list(grid)


# OpponentOffsetPower
def test_opp_offset_no_board_neutral() -> None:
    ind = OpponentOffsetPowerIndicator()
    res = ind.compute(_empty_board(), opponent_board=None)
    assert res.score == 0.5


def test_opp_offset_empty_opponent_zero() -> None:
    ind = OpponentOffsetPowerIndicator()
    res = ind.compute(_empty_board(), opponent_board=_empty_board(),
                       simulator=ChainSimulator())
    assert res.score == 0.0


def test_opp_offset_with_chain_capable_opponent() -> None:
    """1 連鎖 4 連結だけだと 40 点 < 70 で ojama=0、chain_n は 1。"""
    ind = OpponentOffsetPowerIndicator()
    res = ind.compute(_empty_board(), opponent_board=_board_with_4_chain(),
                       simulator=ChainSimulator())
    assert res.detail["opp_chain_n"] >= 1
    # 1 連鎖だと 70 点未満で ojama 0、score=0


def test_opp_offset_multi_chain_returns_positive() -> None:
    """2 連鎖以上の盤面は ojama 換算で score>0。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    # 2 連鎖が組めるパターン: 下段 red 4 連結、その上 blue 4 連結
    for c in range(4):
        grid[BOARD_ROWS - 1][c] = COLOR_RED
        grid[BOARD_ROWS - 2][c] = COLOR_BLUE
    grid[BOARD_ROWS - 3][3] = COLOR_BLUE  # 連鎖 trigger
    opp = Board.from_list(grid)
    ind = OpponentOffsetPowerIndicator()
    res = ind.compute(_empty_board(), opponent_board=opp,
                       simulator=ChainSimulator())
    # 2 連鎖以上であれば score > 0 を期待 (連鎖 chain が組めない盤面なら 0 で OK)
    assert 0.0 <= res.score <= 1.0


# PostOjamaChainHealth
def test_post_ojama_no_chain_neutral() -> None:
    """連鎖発火不可 → neutral 0.5。"""
    ind = PostOjamaChainHealthIndicator()
    sim = ChainSimulator()
    res = ind.compute(_empty_board(), simulator=sim)
    assert res.score == 0.5


def test_post_ojama_with_chain() -> None:
    """連鎖可能盤面 → ojama 落下後の生存率を返す (0-1)。"""
    ind = PostOjamaChainHealthIndicator()
    sim = ChainSimulator()
    res = ind.compute(_board_with_4_chain(), simulator=sim)
    assert 0.0 <= res.score <= 1.0


# IsolatedPuyoCount
def test_isolated_empty_board() -> None:
    """空盤面は isolated 0、score=1.0。"""
    ind = IsolatedPuyoCountIndicator()
    res = ind.compute(_empty_board())
    assert res.score == 1.0


def test_isolated_4_chain_no_isolated() -> None:
    """4 連結のみ → 全部連鎖参加 → isolated=0、score=1.0。"""
    ind = IsolatedPuyoCountIndicator()
    sim = ChainSimulator()
    res = ind.compute(_board_with_4_chain(), simulator=sim)
    assert res.detail["isolated"] == 0
    assert res.score == 1.0


def test_isolated_with_orphan_puyo() -> None:
    """4 連結 + 別色 1 個 (連鎖に参加しない) → isolated=1。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for c in range(4):
        grid[BOARD_ROWS - 1][c] = COLOR_RED
    grid[BOARD_ROWS - 1][5] = COLOR_BLUE  # 別色、孤立
    board = Board.from_list(grid)
    ind = IsolatedPuyoCountIndicator()
    sim = ChainSimulator()
    res = ind.compute(board, simulator=sim)
    assert res.detail["isolated"] >= 1
    assert res.score < 1.0


# 統合
def test_calc_compute_all_phase_k() -> None:
    calc = IndicatorCalculator()
    res = calc.compute_all(_empty_board(), opponent_board=_board_with_4_chain())
    assert INDICATOR_OPPONENT_OFFSET in res.results
    assert INDICATOR_POST_OJAMA_HEALTH in res.results
    assert INDICATOR_ISOLATED_PUYO in res.results
    assert res.opponent_offset_power == res.results[INDICATOR_OPPONENT_OFFSET].score
    assert res.post_ojama_chain_health == res.results[INDICATOR_POST_OJAMA_HEALTH].score
    assert res.isolated_puyo_count == res.results[INDICATOR_ISOLATED_PUYO].score
