"""
hidden_row_tracker.py のテスト

隠し段の物理推論・時系列追跡を検証する。
"""

from __future__ import annotations

import pytest

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_UNKNOWN,
    HIDDEN_ROWS,
    Board,
)
from src.hidden_row_tracker import (
    BoardDiff,
    HiddenRowHypothesis,
    HiddenRowTracker,
)


# ============================
# ヘルパー
# ============================


def make_board(rows_spec: dict[int, dict[int, int]]) -> Board:
    """辞書から Board を生成する (指定外は EMPTY)。"""
    grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
    for row, col_map in rows_spec.items():
        for col, color in col_map.items():
            grid[row][col] = color
    return Board.from_list(grid)


# ============================
# 物理推論
# ============================


class TestPhysicsInference:
    def test_empty_top_visible_all_cols_empty(self):
        """可視最上段が全列空 → 全列確定で empty。"""
        board = make_board({})
        tracker = HiddenRowTracker()
        hyp = tracker.observe(board)
        assert hyp.definitely_empty == set(range(BOARD_COLS))
        assert hyp.possibly_occupied == {}
        assert hyp.confidence == 1.0

    def test_occupied_top_visible_means_unknown(self):
        """可視最上段に puyo → その列は UNKNOWN 扱い。"""
        board = make_board({
            HIDDEN_ROWS: {0: COLOR_RED, 3: COLOR_BLUE},
        })
        hyp = HiddenRowTracker().observe(board)
        assert 0 in hyp.possibly_occupied
        assert 3 in hyp.possibly_occupied
        assert 1 in hyp.definitely_empty
        assert 4 in hyp.definitely_empty

    def test_confidence_ratio(self):
        board = make_board({HIDDEN_ROWS: {0: COLOR_RED, 1: COLOR_BLUE}})
        hyp = HiddenRowTracker().observe(board)
        # 4 列が確定、2 列が UNKNOWN
        assert hyp.confidence == pytest.approx(4 / 6)


# ============================
# 差分計算
# ============================


class TestBoardDiff:
    def test_empty_boards_no_diff(self):
        b1 = make_board({})
        b2 = make_board({})
        diff = HiddenRowTracker().compute_diff(b1, b2)
        assert diff.added_cells == []
        assert diff.removed_cells == []

    def test_new_puyo_added(self):
        b1 = make_board({})
        b2 = make_board({12: {0: COLOR_RED}})
        diff = HiddenRowTracker().compute_diff(b1, b2)
        assert (12, 0, COLOR_RED) in diff.added_cells

    def test_puyo_removed(self):
        b1 = make_board({12: {0: COLOR_RED}})
        b2 = make_board({})
        diff = HiddenRowTracker().compute_diff(b1, b2)
        assert (12, 0, COLOR_RED) in diff.removed_cells


# ============================
# 時系列追跡
# ============================


class TestTemporalTracking:
    def test_first_observation_uses_physics(self):
        tracker = HiddenRowTracker()
        board = make_board({12: {0: COLOR_RED}})
        hyp = tracker.observe(board)
        assert hyp.definitely_empty == set(range(BOARD_COLS))
        # 可視最上段が空なので全列確定

    def test_second_observation_updates_hypothesis(self):
        tracker = HiddenRowTracker()
        # フレーム1: 空盤面
        b1 = make_board({})
        tracker.observe(b1)
        # フレーム2: 可視最上段に puyo 出現 (回し入れの可能性)
        b2 = make_board({HIDDEN_ROWS: {2: COLOR_RED}})
        hyp = tracker.observe(b2)
        assert 2 in hyp.possibly_occupied

    def test_reset_clears_state(self):
        tracker = HiddenRowTracker()
        tracker.observe(make_board({HIDDEN_ROWS: {0: COLOR_RED}}))
        tracker.reset()
        assert tracker._last_board is None
        assert tracker.current_hypothesis().confidence == 0.0


# ============================
# ImageReader との統合
# ============================


class TestImageReaderIntegration:
    """ImageReader 側で既に row 0 推論済みの board を入力としても動作すること。"""

    def test_board_with_unknown_hidden(self):
        """隠し段が UNKNOWN で埋まった Board を観測できる。"""
        grid = [[COLOR_EMPTY] * BOARD_COLS for _ in range(BOARD_ROWS)]
        # ImageReader による推論を模擬
        for col in range(BOARD_COLS):
            grid[0][col] = COLOR_UNKNOWN
        grid[HIDDEN_ROWS][0] = COLOR_RED
        board = Board.from_list(grid)

        hyp = HiddenRowTracker().observe(board)
        assert 0 in hyp.possibly_occupied
        # 他の列は (top_visible が空 + row 0 UNKNOWN) → 可能性占有のまま
