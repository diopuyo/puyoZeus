"""
overlay.py のテスト

AnalysisResult をフレームに合成描画する OverlayRenderer を検証する。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analyzer import Analyzer
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_RED, Board
from src.overlay import (
    INDICATOR_LABELS_JA,
    OverlayRenderer,
    OverlayStyle,
)


# ============================
# ヘルパー
# ============================


def empty_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    return Board.from_list(grid)


def single_erase_board() -> Board:
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for col in range(4):
        grid[12][col] = COLOR_RED
    return Board.from_list(grid)


def make_analysis(b1: Board, b2: Board):
    return Analyzer().analyze_boards(b1, b2)


def make_frame(width: int = 1920, height: int = 1080) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


# ============================
# OverlayStyle
# ============================


class TestOverlayStyle:
    def test_default_flags(self):
        s = OverlayStyle()
        assert s.show_score_bar is True
        assert s.show_indicator_panels is True
        assert s.show_advantage_label is True

    def test_custom_flags(self):
        s = OverlayStyle(show_score_bar=False)
        assert s.show_score_bar is False


# ============================
# OverlayRenderer - render()
# ============================


class TestRender:
    def test_returns_frame_with_same_shape(self):
        result = make_analysis(empty_board(), empty_board())
        frame = make_frame()
        out = OverlayRenderer().render(frame, result)
        assert out.shape == frame.shape

    def test_render_is_non_destructive(self):
        result = make_analysis(empty_board(), empty_board())
        frame = make_frame()
        before = frame.copy()
        _ = OverlayRenderer().render(frame, result)
        assert np.array_equal(frame, before)

    def test_draws_something_on_black_frame(self):
        result = make_analysis(single_erase_board(), empty_board())
        frame = make_frame()
        out = OverlayRenderer().render(frame, result)
        # 何かしら描画されている = 全画素ゼロではない
        assert np.any(out > 0)

    def test_advantage_label_toggle(self):
        result = make_analysis(single_erase_board(), empty_board())
        frame = make_frame()
        with_label = OverlayRenderer(
            OverlayStyle(show_advantage_label=True)
        ).render(frame, result)
        without_label = OverlayRenderer(
            OverlayStyle(show_advantage_label=False)
        ).render(frame, result)
        # ラベル ON のほうが ON 画素が多いはず
        assert np.sum(with_label > 0) > np.sum(without_label > 0)

    def test_disable_all_panels_minimizes_drawing(self):
        result = make_analysis(single_erase_board(), empty_board())
        frame = make_frame()
        full = OverlayRenderer().render(frame, result)
        minimal = OverlayRenderer(
            OverlayStyle(
                show_score_bar=False,
                show_indicator_panels=False,
                show_advantage_label=False,
            )
        ).render(frame, result)
        assert np.sum(minimal > 0) < np.sum(full > 0)


# ============================
# OverlayRenderer - render_transparent()
# ============================


class TestRenderTransparent:
    def test_returns_bgra(self):
        result = make_analysis(empty_board(), empty_board())
        out = OverlayRenderer().render_transparent(800, 600, result)
        assert out.shape == (600, 800, 4)
        assert out.dtype == np.uint8

    def test_alpha_has_nonzero(self):
        result = make_analysis(single_erase_board(), empty_board())
        out = OverlayRenderer().render_transparent(1280, 720, result)
        # アルファチャンネルに何か描画されていること
        assert np.any(out[..., 3] > 0)

    def test_edges_are_transparent(self):
        result = make_analysis(empty_board(), empty_board())
        out = OverlayRenderer().render_transparent(1280, 720, result)
        # 右下の隅は何も描画されない想定 → alpha=0
        assert out[-1, -1, 3] == 0


# ============================
# ラベル定数
# ============================


class TestLabels:
    def test_all_indicators_have_japanese_labels(self):
        from src.old.indicators import ALL_INDICATOR_NAMES

        for name in ALL_INDICATOR_NAMES:
            assert name in INDICATOR_LABELS_JA
