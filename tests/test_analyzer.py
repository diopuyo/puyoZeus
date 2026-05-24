"""
analyzer.py のテスト

統合エンドポイントが盤面/フレームから完全な AnalysisResult を返すことを検証する。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_RED, Board
from src.image_reader import BoardRegion, ImageReader
from src.analyzer import AnalysisResult, Analyzer, PlayerAnalysis


# ============================
# ヘルパー
# ============================


def empty_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(grid)


def single_erase_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    grid[12][0] = COLOR_RED
    grid[12][1] = COLOR_RED
    grid[12][2] = COLOR_RED
    grid[12][3] = COLOR_RED
    return Board.from_list(grid)


def make_dummy_frame() -> np.ndarray:
    """真っ黒な 1080p フレーム (盤面として読めば全て空)。"""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


# ============================
# PlayerAnalysis
# ============================


class TestPlayerAnalysis:
    def test_to_dict_structure(self):
        analyzer = Analyzer()
        p = analyzer.analyze_player(single_erase_board())
        d = p.to_dict()
        assert "board" in d
        assert "chain_count" in d
        assert "indicators" in d
        assert d["chain_count"] == 1


# ============================
# AnalysisResult
# ============================


class TestAnalysisResult:
    def test_to_dict_has_players_and_score(self):
        analyzer = Analyzer()
        result = analyzer.analyze_boards(empty_board(), empty_board())
        d = result.to_dict()
        assert "player1" in d
        assert "player2" in d
        assert "score" in d

    def test_timestamp_included_when_provided(self):
        analyzer = Analyzer()
        result = analyzer.analyze_boards(
            empty_board(), empty_board(), timestamp=1.5
        )
        assert result.timestamp == 1.5
        assert result.to_dict()["timestamp"] == 1.5

    def test_timestamp_omitted_when_none(self):
        analyzer = Analyzer()
        result = analyzer.analyze_boards(empty_board(), empty_board())
        assert result.timestamp is None
        assert "timestamp" not in result.to_dict()


# ============================
# Analyzer - 盤面入力
# ============================


class TestAnalyzeBoards:
    def test_returns_analysis_result(self):
        analyzer = Analyzer()
        result = analyzer.analyze_boards(empty_board(), empty_board())
        assert isinstance(result, AnalysisResult)

    def test_equal_boards_yield_even_score(self):
        analyzer = Analyzer()
        result = analyzer.analyze_boards(empty_board(), empty_board())
        assert result.score.total_score == 0.0

    def test_player1_has_chain_advantage(self):
        analyzer = Analyzer()
        result = analyzer.analyze_boards(single_erase_board(), empty_board())
        # 1P が連鎖可能 → スコアは正
        assert result.score.total_score > 0

    def test_both_players_analyzed(self):
        analyzer = Analyzer()
        result = analyzer.analyze_boards(
            single_erase_board(), single_erase_board()
        )
        assert result.player1.chain_result.chain_count == 1
        assert result.player2.chain_result.chain_count == 1
        # 同条件なのでスコアは 0
        assert result.score.total_score == 0.0


# ============================
# Analyzer - フレーム入力
# ============================


class TestAnalyzeFrame:
    def test_returns_result_for_black_frame(self):
        analyzer = Analyzer()
        result = analyzer.analyze_frame(make_dummy_frame())
        assert isinstance(result, AnalysisResult)
        # 真っ黒=両盤面とも空 → スコア 0
        assert result.score.total_score == 0.0

    def test_timestamp_propagated(self):
        analyzer = Analyzer()
        result = analyzer.analyze_frame(make_dummy_frame(), timestamp=2.0)
        assert result.timestamp == 2.0


# ============================
# Analyzer - 差し替え可能性
# ============================


class TestAnalyzerComposition:
    def test_custom_image_reader_used(self):
        # 小さいリージョンを渡した ImageReader でも動作する
        region = BoardRegion(x=0, y=0, width=60, height=130)
        reader = ImageReader(p1_region=region, p2_region=region)
        analyzer = Analyzer(image_reader=reader)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = analyzer.analyze_frame(frame)
        assert isinstance(result, AnalysisResult)

    def test_analyze_player_only(self):
        analyzer = Analyzer()
        p = analyzer.analyze_player(single_erase_board())
        assert isinstance(p, PlayerAnalysis)
        assert p.chain_result.chain_count == 1
