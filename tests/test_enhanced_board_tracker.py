"""V2.4 EnhancedBoardTracker のテスト。"""
from __future__ import annotations

from src.board import (
    BOARD_COLS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
    Board,
)
from src.enhanced_board_tracker import (
    EnhancedBoardTracker,
    EnhancedTrackingStats,
)


def _make_board(cells: dict[tuple[int, int], int]) -> Board:
    b = Board()
    for (row, col), color in cells.items():
        b.set(row, col, color)
    return b


def test_initial_update_accepts_observation() -> None:
    """初回 update は観測をそのまま採用 (StatefulBoardTracker の挙動を継承)。"""
    tracker = EnhancedBoardTracker()
    obs = _make_board({(12, 2): COLOR_RED})
    result = tracker.update(obs)
    assert result.get(12, 2) == COLOR_RED


def test_next_link_correction_applied() -> None:
    """V2.1: 直前 next_pair が (RED, BLUE) で観測が (RED, PURPLE) → BLUE に補正。"""
    tracker = EnhancedBoardTracker()
    # 1 回目: ブート + next_pair=(RED, BLUE) を保存
    tracker.update(Board(), next_pair=(COLOR_RED, COLOR_BLUE))
    # 2 回目: 新規 2 セル出現で色不一致
    obs = _make_board({(11, 2): COLOR_RED, (12, 2): COLOR_PURPLE})
    result = tracker.update(obs, next_pair=None)
    # PURPLE は BLUE に補正されているはず
    assert result.get(11, 2) == COLOR_RED
    assert result.get(12, 2) == COLOR_BLUE
    assert tracker.last_stats.next_link_corrected == 1


def test_connectivity_correction_applied() -> None:
    """V2.3: 孤立異色セルが多数色に補正される。"""
    tracker = EnhancedBoardTracker()
    # 1 回目: ブート (空盤面)
    tracker.update(Board())
    # 2 回目: 赤に囲まれた紫 1 セル
    cells = {
        (5, 2): COLOR_RED,
        (7, 2): COLOR_RED,
        (6, 1): COLOR_RED,
        (6, 3): COLOR_RED,
        (6, 2): COLOR_PURPLE,
    }
    obs = _make_board(cells)
    result = tracker.update(obs)
    # 紫が赤に補正されているはず (V2.3)
    # ただし StatefulBoardTracker が浮遊削除する場合あり、まずは V2.3 補正の発生をチェック
    assert tracker.last_stats.connectivity_corrected >= 1


def test_pair_inconsistent_logged() -> None:
    """V2.2: 1 セル新規出現で pair_inconsistent フラグが立つ。"""
    tracker = EnhancedBoardTracker()
    tracker.update(Board())
    obs = _make_board({(12, 2): COLOR_RED})
    tracker.update(obs)
    # 1 セルだけ新規 → 不整合検出
    assert tracker.last_stats.pair_inconsistent == 1


def test_pair_consistent_when_two_new_cells() -> None:
    """2 セル新規出現は整合 → pair_inconsistent=0。"""
    tracker = EnhancedBoardTracker()
    tracker.update(Board())
    obs = _make_board({(11, 2): COLOR_RED, (12, 2): COLOR_BLUE})
    tracker.update(obs)
    assert tracker.last_stats.pair_inconsistent == 0


def test_reset_clears_state() -> None:
    """reset() で内部状態が消える。"""
    tracker = EnhancedBoardTracker()
    tracker.update(_make_board({(12, 2): COLOR_RED}))
    assert tracker.initialized is True
    tracker.reset()
    assert tracker.initialized is False
    # 次の update も初期化扱い
    tracker.update(_make_board({(12, 3): COLOR_BLUE}))
    assert tracker.current.get(12, 3) == COLOR_BLUE


def test_stats_object_returned() -> None:
    """last_stats フィールド構造の確認。"""
    tracker = EnhancedBoardTracker()
    tracker.update(Board())
    s = tracker.last_stats
    assert isinstance(s, EnhancedTrackingStats)
    assert s.next_link_corrected == 0
    assert s.connectivity_corrected == 0
    assert s.pair_inconsistent == 0


def test_next_pair_propagation() -> None:
    """update に渡す next_pair が次フレームの prev_next_pair になる。"""
    tracker = EnhancedBoardTracker()
    # ブート + next_pair=(RED, GRN) 保存
    tracker.update(Board(), next_pair=(COLOR_RED, COLOR_GREEN))
    # 新規 2 セル: (RED, YEL) 観測
    obs = _make_board({(11, 2): COLOR_RED, (12, 2): COLOR_YELLOW})
    result = tracker.update(obs, next_pair=(COLOR_BLUE, COLOR_PURPLE))
    # 直前 next_pair=(RED, GRN) で YEL → GRN に補正
    assert result.get(11, 2) == COLOR_RED
    assert result.get(12, 2) == COLOR_GREEN
