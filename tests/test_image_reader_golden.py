"""
ImageReader ゴールデンテスト

合成フレーム (tests/fixtures.py で生成) から Board を読み取り、
入力 Board と完全一致することを検証する。

Board → image → Board の往復整合性保証により、画像認識コードの
リグレッションを早期検出する。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    HIDDEN_ROWS,
    Board,
)


def _apply_hidden_inference(board: Board) -> Board:
    """
    ImageReader の物理推論と同じロジックを適用。
    可視最上段が空なら row 0 = EMPTY、非空なら row 0 = UNKNOWN。
    """
    new_board = board.copy()
    top_visible = HIDDEN_ROWS
    for col in range(BOARD_COLS):
        top_cell = new_board.get(top_visible, col)
        for hidden_row in range(HIDDEN_ROWS):
            if top_cell == COLOR_EMPTY:
                new_board.set(hidden_row, col, COLOR_EMPTY)
            else:
                new_board.set(hidden_row, col, COLOR_UNKNOWN)
    return new_board
from src.image_reader import ImageReader

from tests.fixtures import (
    make_synthetic_frame,
    sample_4_chain_board,
    sample_all_colors_board,
    sample_stacked_board,
)


# ============================
# 単純な色別ラウンドトリップ
# ============================


class TestColorRoundtrip:
    """全色の単一配置が正しく読み取れることを保証する。"""

    @pytest.mark.parametrize(
        "color",
        [COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA],
    )
    def test_single_color_at_row12_col0(self, color: int):
        grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
        grid[12][0] = color
        board = Board.from_list(grid)

        frame = make_synthetic_frame(board_1p=board)
        result = ImageReader().read_board(
            frame,
            region=ImageReader()._p1_region,
        )
        assert result.get(12, 0) == color, f"色 {color} の往復が壊れた"

    def test_empty_board_reads_as_empty(self):
        empty = Board.from_list(
            [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
        )
        frame = make_synthetic_frame(board_1p=empty)
        result = ImageReader().read_board(
            frame, region=ImageReader()._p1_region
        )
        # 隠し段は UNKNOWN として読まれる
        assert result == _apply_hidden_inference(empty)


# ============================
# 複雑な盤面のラウンドトリップ
# ============================


class TestBoardRoundtrip:
    def test_all_colors_roundtrip(self):
        board = sample_all_colors_board()
        frame = make_synthetic_frame(board_1p=board)
        read = ImageReader().read_board(
            frame, region=ImageReader()._p1_region
        )
        assert read == _apply_hidden_inference(board)

    def test_stacked_roundtrip(self):
        board = sample_stacked_board()
        frame = make_synthetic_frame(board_1p=board)
        read = ImageReader().read_board(
            frame, region=ImageReader()._p1_region
        )
        assert read == _apply_hidden_inference(board)

    def test_4_chain_roundtrip(self):
        board = sample_4_chain_board()
        frame = make_synthetic_frame(board_1p=board)
        read = ImageReader().read_board(
            frame, region=ImageReader()._p1_region
        )
        assert read == _apply_hidden_inference(board)


# ============================
# 両プレイヤー同時読み取り
# ============================


class TestBothPlayersRoundtrip:
    def test_both_sides_independent(self):
        b1 = sample_all_colors_board()
        b2 = sample_4_chain_board()
        frame = make_synthetic_frame(board_1p=b1, board_2p=b2)
        reader = ImageReader()
        read_1p, read_2p = reader.read_both_boards(frame)
        assert read_1p == _apply_hidden_inference(b1)
        assert read_2p == _apply_hidden_inference(b2)


# ============================
# 退行テスト (Phase 2 の連鎖結果を維持する)
# ============================


class TestChainContinuity:
    """画像経由でも連鎖シミュレーションが同じ結果になること。"""

    def test_chain_count_preserved_through_image(self):
        from src.chain import ChainSimulator

        board = sample_4_chain_board()
        direct = ChainSimulator().simulate(board).chain_count

        frame = make_synthetic_frame(board_1p=board)
        read = ImageReader().read_board(
            frame, region=ImageReader()._p1_region
        )
        via_image = ChainSimulator().simulate(read).chain_count
        assert via_image == direct == 4
