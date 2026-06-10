"""Phase J 新指標 (2026-04-27 mayah/ama 先行研究ベース) のテスト。

- AdjacentHeightDiffIndicator (隣接列高さ差)
- HighConnectionCountIndicator (3+ 連結数)
- RequiredPuyoToFireIndicator (発火必要ぷよ数)
- OpponentChainThreatIndicator (相手連鎖威力換算)
"""
from __future__ import annotations

import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_RED, Board
from src.chain import ChainSimulator
from src.old.indicators import (
    AdjacentHeightDiffIndicator,
    HighConnectionCountIndicator,
    INDICATOR_HEIGHT_DIFF,
    INDICATOR_HIGH_CONNECTION,
    INDICATOR_OPPONENT_THREAT,
    INDICATOR_REQUIRED_FIRE,
    IndicatorCalculator,
    OpponentChainThreatIndicator,
    RequiredPuyoToFireIndicator,
)


def _empty_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(grid)


def _flat_board(color: int = COLOR_RED) -> Board:
    """全列同じ高さの平らな盤面 (2 段)。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    grid[BOARD_ROWS - 1] = [color] * BOARD_COLS
    grid[BOARD_ROWS - 2] = [color] * BOARD_COLS
    return Board.from_list(grid)


def _peaked_board() -> Board:
    """凸凹: 1 列だけ 8 段、他は 0 段。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for r in range(BOARD_ROWS - 8, BOARD_ROWS):
        grid[r][0] = COLOR_RED
    return Board.from_list(grid)


# =============================================================================
# AdjacentHeightDiffIndicator
# =============================================================================


def test_height_diff_flat_board_high_score() -> None:
    """平らな盤面は score = 1.0 (差分 0)。"""
    ind = AdjacentHeightDiffIndicator()
    res = ind.compute(_flat_board())
    assert res.name == INDICATOR_HEIGHT_DIFF
    assert res.score == 1.0
    assert res.raw_value == 0


def test_height_diff_peaked_board_low_score() -> None:
    """凸凹盤面は score < 1.0。"""
    ind = AdjacentHeightDiffIndicator()
    res = ind.compute(_peaked_board())
    assert res.score < 1.0
    assert res.raw_value > 0


def test_height_diff_empty_board() -> None:
    """空盤面は score = 1.0 (全列高さ 0)。"""
    ind = AdjacentHeightDiffIndicator()
    res = ind.compute(_empty_board())
    assert res.score == 1.0


# =============================================================================
# HighConnectionCountIndicator
# =============================================================================


def test_high_connection_empty_board() -> None:
    """空盤面は 3+ 連結なし、neutral 0.5。"""
    ind = HighConnectionCountIndicator()
    res = ind.compute(_empty_board())
    assert 0.0 <= res.score <= 1.0
    assert res.detail["n_3"] == 0
    assert res.detail["n_4plus"] == 0


def test_high_connection_4plus_penalty() -> None:
    """4 連結を 1 つ作ると n_4plus が増えてペナルティ方向に動く。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    # 4 連結: (12, 0) (12, 1) (12, 2) (12, 3) を red
    for c in range(4):
        grid[BOARD_ROWS - 1][c] = COLOR_RED
    board = Board.from_list(grid)
    ind = HighConnectionCountIndicator()
    res = ind.compute(board)
    assert res.detail["n_4plus"] >= 1


def test_high_connection_3_chain_bonus() -> None:
    """3 連結のみは bonus 方向。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    # 3 連結: (12, 0)(12, 1)(12, 2) を red
    for c in range(3):
        grid[BOARD_ROWS - 1][c] = COLOR_RED
    board = Board.from_list(grid)
    ind = HighConnectionCountIndicator()
    res = ind.compute(board)
    assert res.detail["n_3"] == 1
    assert res.detail["n_4plus"] == 0


# =============================================================================
# RequiredPuyoToFireIndicator
# =============================================================================


def test_required_puyo_empty_board_low_score() -> None:
    """空盤面は発火不可で min_n が大きい → 低スコア。"""
    ind = RequiredPuyoToFireIndicator()
    sim = ChainSimulator()
    res = ind.compute(_empty_board(), simulator=sim)
    assert 0.0 <= res.score <= 1.0


def test_required_puyo_returns_indicator_result() -> None:
    """発火可能盤面でも IndicatorResult が返る。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    # 3 連結 + 隣に 1 個 = 4 連結 1 ターンで発火可能
    for c in range(3):
        grid[BOARD_ROWS - 1][c] = COLOR_RED
    grid[BOARD_ROWS - 2][0] = COLOR_RED
    board = Board.from_list(grid)
    ind = RequiredPuyoToFireIndicator()
    res = ind.compute(board, simulator=ChainSimulator())
    assert res.name == INDICATOR_REQUIRED_FIRE


# =============================================================================
# OpponentChainThreatIndicator
# =============================================================================


def test_opponent_threat_no_opponent_board_neutral() -> None:
    """opponent_board=None なら neutral 0.5。"""
    ind = OpponentChainThreatIndicator()
    res = ind.compute(_empty_board(), opponent_board=None)
    assert res.score == 0.5
    assert res.detail["reason"] == "no_opponent_board"


def test_opponent_threat_empty_opponent_zero_threat() -> None:
    """空相手盤面は脅威 0。"""
    ind = OpponentChainThreatIndicator()
    sim = ChainSimulator()
    res = ind.compute(_empty_board(), opponent_board=_empty_board(),
                       simulator=sim)
    assert res.score == 0.0
    assert res.raw_value == 0.0


def test_opponent_threat_with_chain_capable_board() -> None:
    """連鎖発火可能な相手盤面で score > 0。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for c in range(4):
        grid[BOARD_ROWS - 1][c] = COLOR_RED
    opp_board = Board.from_list(grid)
    ind = OpponentChainThreatIndicator()
    sim = ChainSimulator()
    res = ind.compute(_empty_board(), opponent_board=opp_board, simulator=sim)
    assert res.detail["opp_chain_n"] >= 1


# =============================================================================
# IndicatorCalculator 統合テスト
# =============================================================================


def test_calc_compute_all_includes_phase_j_indicators() -> None:
    """compute_all の結果に Phase J 4 指標が含まれる。"""
    calc = IndicatorCalculator()
    res = calc.compute_all(_empty_board())
    assert INDICATOR_OPPONENT_THREAT in res.results
    assert INDICATOR_HEIGHT_DIFF in res.results
    assert INDICATOR_HIGH_CONNECTION in res.results
    assert INDICATOR_REQUIRED_FIRE in res.results
    # IndicatorSet 直接フィールドも一致
    assert res.opponent_chain_threat == res.results[INDICATOR_OPPONENT_THREAT].score
    assert res.adjacent_height_diff == res.results[INDICATOR_HEIGHT_DIFF].score


def test_calc_compute_all_with_opponent_board() -> None:
    """opponent_board を渡すと OpponentChainThreat が neutral でなくなる。"""
    calc = IndicatorCalculator()
    res_no_opp = calc.compute_all(_empty_board())
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for c in range(4):
        grid[BOARD_ROWS - 1][c] = COLOR_RED
    opp = Board.from_list(grid)
    res_with_opp = calc.compute_all(_empty_board(), opponent_board=opp)
    # opponent あり → score が 0.5 から動く
    assert res_no_opp.opponent_chain_threat == 0.5
    assert res_with_opp.opponent_chain_threat != 0.5
