"""
ネクスト/ダブルネクスト受け入れ余地指標 (next_acceptance) のテスト。

IndicatorCalculator.compute_all() の next_pair / dnext_pair 引数と
IndicatorSet.next_acceptance フィールドの挙動を検証する。
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
from src.indicators import (
    INDICATOR_NEXT_ACCEPTANCE,
    NEXT_ACCEPTANCE_NEUTRAL,
    IndicatorCalculator,
    IndicatorSet,
    _compute_next_acceptance,
    _place_pair,
)
from src.chain import ChainSimulator


# ============================
# テスト用ヘルパー
# ============================


def empty_grid() -> list[list[int]]:
    """13×6 の全空グリッド。"""
    return [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]


def make_empty_board() -> Board:
    return Board.from_list(empty_grid())


def make_almost_chain_board() -> Board:
    """
    赤を 1 つ置けば横 4 連結が成立する盤面 (R R R _ _ _)。

    最下段に赤 3 つだけ並んでおり、ネクストで赤を追加すれば連鎖が起きる。
    """
    grid = empty_grid()
    grid[12][0] = COLOR_RED
    grid[12][1] = COLOR_RED
    grid[12][2] = COLOR_RED
    return Board.from_list(grid)


def make_filled_board() -> Board:
    """
    ほぼ満杯で puyo を置けない盤面 (placement 不可テスト用)。

    全列を 13 段まで赤で埋める。is_dead() が True になる。
    """
    grid = [[COLOR_RED] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(grid)


# ============================
# IndicatorSet.next_acceptance フィールド
# ============================


class TestIndicatorSetNextAcceptance:
    def test_default_is_neutral(self):
        """next_acceptance のデフォルトは中立値 0.5。"""
        s = IndicatorSet(results={})
        assert s.next_acceptance == NEXT_ACCEPTANCE_NEUTRAL
        assert s.next_acceptance == 0.5

    def test_explicit_value(self):
        """明示的に next_acceptance を渡せば反映される。"""
        s = IndicatorSet(results={}, next_acceptance=0.8)
        assert s.next_acceptance == 0.8


# ============================
# IndicatorCalculator.compute_all
# ============================


class TestComputeAllNext:
    def test_no_next_pair_neutral(self):
        """next_pair / dnext_pair 未指定なら中立値 0.5 を返す。"""
        calc = IndicatorCalculator()
        result = calc.compute_all(make_empty_board())
        assert result.next_acceptance == NEXT_ACCEPTANCE_NEUTRAL

    def test_only_next_given_neutral(self):
        """片方のみなら中立値 (両方揃わなければ評価しない)。"""
        calc = IndicatorCalculator()
        result = calc.compute_all(
            make_empty_board(),
            next_pair=(COLOR_RED, COLOR_BLUE),
            dnext_pair=None,
        )
        assert result.next_acceptance == NEXT_ACCEPTANCE_NEUTRAL

    def test_both_given_in_range(self):
        """両方与えれば 0.0〜1.0 の範囲のスコアを返す。"""
        calc = IndicatorCalculator()
        result = calc.compute_all(
            make_almost_chain_board(),
            next_pair=(COLOR_RED, COLOR_RED),
            dnext_pair=(COLOR_BLUE, COLOR_BLUE),
        )
        assert 0.0 <= result.next_acceptance <= 1.0

    def test_chain_extending_pair_above_zero(self):
        """連鎖を伸ばせる next/dnext 配置で next_acceptance > 0 になる。"""
        calc = IndicatorCalculator()
        # 赤を追加すれば 4 連結成立で 1 連鎖発生する盤面
        result = calc.compute_all(
            make_almost_chain_board(),
            next_pair=(COLOR_RED, COLOR_YELLOW),
            dnext_pair=(COLOR_GREEN, COLOR_BLUE),
        )
        assert result.next_acceptance > 0.0

    def test_compute_all_preserves_existing_indicators(self):
        """既存 8 指標 + 拡張 4 指標が同時に計算される (互換性確認)。"""
        from src.indicators import ALL_INDICATOR_NAMES
        calc = IndicatorCalculator()
        result = calc.compute_all(
            make_almost_chain_board(),
            next_pair=(COLOR_RED, COLOR_RED),
            dnext_pair=(COLOR_BLUE, COLOR_BLUE),
        )
        # ALL_INDICATOR_NAMES (8 メイン指標) は必ず含まれる
        for name in ALL_INDICATOR_NAMES:
            assert name in result.results
        # 拡張 4 指標も含まれるため >= 12
        assert len(result.results) >= 8


# ============================
# _compute_next_acceptance ロジック
# ============================


class TestComputeNextAcceptanceFunc:
    def test_irrelevant_pair_returns_zero(self):
        """連鎖に寄与しない色 (青/黄) では伸長 0、score=0。"""
        sim = ChainSimulator()
        score, detail = _compute_next_acceptance(
            make_almost_chain_board(),
            next_pair=(COLOR_BLUE, COLOR_YELLOW),
            dnext_pair=(COLOR_GREEN, COLOR_YELLOW),
            simulator=sim,
        )
        assert score == 0.0
        assert detail["delta"] == 0

    def test_chain_extending_pair_returns_positive(self):
        """赤 puyo を追加できる pair で連鎖が伸びる。"""
        sim = ChainSimulator()
        score, detail = _compute_next_acceptance(
            make_almost_chain_board(),
            next_pair=(COLOR_RED, COLOR_RED),
            dnext_pair=(COLOR_RED, COLOR_RED),
            simulator=sim,
        )
        # 赤を 1 列目に置けば 4 連結成立 → chain >= 1
        assert score > 0.0
        assert detail["best_chain"] > detail["base_chain"]

    def test_score_clamped_to_one(self):
        """delta が大きくても score は 1.0 を超えない。"""
        sim = ChainSimulator()
        score, _ = _compute_next_acceptance(
            make_almost_chain_board(),
            next_pair=(COLOR_RED, COLOR_RED),
            dnext_pair=(COLOR_RED, COLOR_RED),
            simulator=sim,
        )
        assert score <= 1.0


# ============================
# _place_pair ヘルパー
# ============================


class TestPlacePair:
    def test_vertical_rotation_0(self):
        """rotation=0: 縦配置で TOP が上、BOT が下。"""
        board = make_empty_board()
        result = _place_pair(board, (COLOR_RED, COLOR_BLUE), col=0, rotation=0)
        assert result is not None
        # 最下段が BOT(青)、その上が TOP(赤)
        assert result.get(BOARD_ROWS - 1, 0) == COLOR_BLUE
        assert result.get(BOARD_ROWS - 2, 0) == COLOR_RED

    def test_horizontal_rotation_1(self):
        """rotation=1: 横配置で TOP が左、BOT が右。"""
        board = make_empty_board()
        result = _place_pair(board, (COLOR_RED, COLOR_BLUE), col=2, rotation=1)
        assert result is not None
        assert result.get(BOARD_ROWS - 1, 2) == COLOR_RED
        assert result.get(BOARD_ROWS - 1, 3) == COLOR_BLUE

    def test_invalid_column_horizontal(self):
        """横配置で col=5 は右隣がないので None。"""
        board = make_empty_board()
        result = _place_pair(board, (COLOR_RED, COLOR_BLUE), col=5, rotation=1)
        assert result is None

    def test_full_column_returns_none(self):
        """満杯盤面では配置不可で None。"""
        board = make_filled_board()
        result = _place_pair(board, (COLOR_RED, COLOR_BLUE), col=0, rotation=0)
        assert result is None

    def test_does_not_mutate_input_board(self):
        """元の盤面は変更されない (copy ベース)。"""
        board = make_empty_board()
        _place_pair(board, (COLOR_RED, COLOR_BLUE), col=0, rotation=0)
        # 元盤面は依然空のまま
        assert board.get(BOARD_ROWS - 1, 0) == COLOR_EMPTY


# ============================
# Scorer 連携
# ============================


class TestScorerIntegration:
    def test_default_weights_includes_next_acceptance(self):
        """Scorer の DEFAULT_WEIGHTS に next_acceptance が含まれる。"""
        from src.scorer import DEFAULT_WEIGHTS
        assert INDICATOR_NEXT_ACCEPTANCE in DEFAULT_WEIGHTS
        assert DEFAULT_WEIGHTS[INDICATOR_NEXT_ACCEPTANCE] == 0.6

    def test_scorer_handles_next_acceptance(self):
        """Scorer は next_acceptance の差分を total_score に反映する。"""
        from src.scorer import Scorer
        s = Scorer()
        # 1P が next_acceptance 高、2P が低
        p1 = IndicatorSet(results={}, next_acceptance=1.0)
        p2 = IndicatorSet(results={}, next_acceptance=0.0)
        result = s.score(p1, p2)
        # 正の重みで 1P 有利
        assert result.total_score > 0

    def test_scorer_breakdown_excludes_next_acceptance(self):
        """next_acceptance は breakdown には含まれない (互換性確保)。"""
        from src.indicators import ALL_INDICATOR_NAMES
        from src.scorer import Scorer
        s = Scorer()
        p1 = IndicatorSet(results={}, next_acceptance=0.5)
        p2 = IndicatorSet(results={}, next_acceptance=0.5)
        result = s.score(p1, p2)
        assert INDICATOR_NEXT_ACCEPTANCE not in result.player1_breakdown
        assert set(result.player1_breakdown.keys()) == set(ALL_INDICATOR_NAMES)
