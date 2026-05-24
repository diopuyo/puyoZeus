"""TemporalSmoother の単体テスト。"""
from __future__ import annotations

import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_RED,
    Board,
)
from src.temporal_smoother import TemporalSmoother


def _board_with(row: int, col: int, color: int) -> Board:
    """指定セルだけ色を置いた盤面を作る (他は空)。"""
    board = Board()
    board.set(row, col, color)
    return board


def _all_color_board(color: int) -> Board:
    """全セル同じ色の盤面。"""
    board = Board()
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            board.set(r, c, color)
    return board


class TestBasicBehavior:
    def test_single_frame_passthrough(self) -> None:
        """window_size=1 なら入力そのままが返る。"""
        sm = TemporalSmoother(window_size=1)
        board = _board_with(5, 3, COLOR_RED)
        out = sm.update(board)
        assert out.get(5, 3) == COLOR_RED
        assert out.get(0, 0) == COLOR_EMPTY

    def test_history_fill(self) -> None:
        """履歴が window_size に達していなくてもエラーなく動く。"""
        sm = TemporalSmoother(window_size=5)
        board = _board_with(6, 2, COLOR_BLUE)
        out = sm.update(board)
        assert out.get(6, 2) == COLOR_BLUE
        assert len(sm) == 1

    def test_reset_clears_history(self) -> None:
        sm = TemporalSmoother(window_size=3)
        sm.update(_board_with(0, 0, COLOR_RED))
        sm.update(_board_with(0, 0, COLOR_RED))
        assert len(sm) == 2
        sm.reset()
        assert len(sm) == 0

    def test_invalid_window_size(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            TemporalSmoother(window_size=0)
        with pytest.raises(ValueError):
            TemporalSmoother(window_size=-1)


class TestNoiseRejection:
    """UI/halo ノイズを単独フレームで打ち消せるか。"""

    def test_single_noise_frame_rejected(self) -> None:
        """
        window=15 で 14 フレーム赤、1 フレームだけ緑 (halo 誤認)
        → 多数決で赤が残る。
        """
        sm = TemporalSmoother(window_size=15)
        red_board = _board_with(5, 3, COLOR_RED)
        green_noise = _board_with(5, 3, COLOR_GREEN)

        # 14 フレーム赤 → 出力は赤のまま
        for _ in range(14):
            sm.update(red_board)
        # 15 フレーム目: halo 誤認で緑
        out = sm.update(green_noise)
        assert out.get(5, 3) == COLOR_RED, "1 フレのノイズは多数決で排除されるべき"

    def test_multi_frame_noise_rejected(self) -> None:
        """
        window=15 で 10 フレーム赤、5 フレーム他色混入 → まだ赤優勢。
        """
        sm = TemporalSmoother(window_size=15)
        for _ in range(10):
            sm.update(_board_with(5, 3, COLOR_RED))
        for _ in range(5):
            sm.update(_board_with(5, 3, COLOR_GREEN))
        # 多数決: 赤 10, 緑 5 → 赤が勝つ
        # 最終盤面は最後に追加した緑だが、出力は集計結果
        out = sm.update(_board_with(5, 3, COLOR_GREEN))
        # 履歴: 赤 10, 緑 6 → 赤が勝つ
        assert out.get(5, 3) == COLOR_RED

    def test_mode_takes_majority(self) -> None:
        """3 色が混在した場合、最多色が勝つ。"""
        sm = TemporalSmoother(window_size=10)
        for _ in range(4):
            sm.update(_board_with(1, 1, COLOR_BLUE))
        for _ in range(3):
            sm.update(_board_with(1, 1, COLOR_GREEN))
        for _ in range(3):
            sm.update(_board_with(1, 1, COLOR_RED))
        out = sm.update(_board_with(1, 1, COLOR_BLUE))
        # 履歴: 青 5, 緑 3, 赤 3 → 青勝ち
        assert out.get(1, 1) == COLOR_BLUE


class TestSlidingWindow:
    def test_window_forgets_old_frames(self) -> None:
        """
        window=3 なら古いフレームは忘れる: 赤×3 → 緑×3 で緑に変わる。
        """
        sm = TemporalSmoother(window_size=3)
        red = _board_with(2, 2, COLOR_RED)
        green = _board_with(2, 2, COLOR_GREEN)

        # 赤で満たす
        for _ in range(3):
            sm.update(red)
        out = sm.update(red)  # まだ赤
        assert out.get(2, 2) == COLOR_RED

        # 緑で 3 回上書き → 履歴は緑のみに
        for _ in range(2):
            sm.update(green)
        out = sm.update(green)
        assert out.get(2, 2) == COLOR_GREEN

    def test_real_color_change_follows_with_delay(self) -> None:
        """
        連鎖消去のような本物の色→空変化は、window 内の多数派が変わる
        タイミングで追従する (delayed の特性)。
        """
        sm = TemporalSmoother(window_size=5)
        red = _board_with(4, 4, COLOR_RED)
        empty = Board()  # 全空

        # 5 フレーム赤
        for _ in range(5):
            sm.update(red)
        # 消去: 3 フレーム空を入れる → 履歴: 赤 2, 空 3 → 空勝ち
        sm.update(empty)
        sm.update(empty)
        out = sm.update(empty)
        assert out.get(4, 4) == COLOR_EMPTY


class TestCellIndependence:
    """セルごとに独立に集計されること。"""

    def test_different_cells_dont_interfere(self) -> None:
        sm = TemporalSmoother(window_size=5)
        for _ in range(5):
            board = Board()
            board.set(3, 3, COLOR_RED)
            board.set(4, 4, COLOR_BLUE)
            sm.update(board)
        out = sm.update(Board())  # 全空 1 回追加
        # (3,3) 赤 5, 空 1 → 赤 // (4,4) 青 5, 空 1 → 青
        # window=5 なのでリングバッファは直近 5 件
        # 追加後: (3,3) 赤 4, 空 1 → 赤 // (4,4) 青 4, 空 1 → 青
        assert out.get(3, 3) == COLOR_RED
        assert out.get(4, 4) == COLOR_BLUE


class TestOjamaAndEmpty:
    def test_ojama_majority_survives(self) -> None:
        """お邪まは色コード 9 だが bincount の minlength=10 で対応されている。"""
        sm = TemporalSmoother(window_size=3)
        ojama = _board_with(0, 0, COLOR_OJAMA)
        for _ in range(3):
            sm.update(ojama)
        out = sm.update(Board())  # 空 1 投入
        # 履歴: お邪ま 2, 空 1 → お邪まが残る
        assert out.get(0, 0) == COLOR_OJAMA
