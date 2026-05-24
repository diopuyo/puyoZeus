"""
src/ojama_tracker.py のテスト
"""
from __future__ import annotations

import pytest

from src.ojama_tracker import (
    SIDE_P1,
    SIDE_P2,
    OjamaTimelineEvent,
    OjamaTimelineTracker,
)
from src.ojama_warning import (
    ICON_ROCK,
    ICON_SMALL,
    OjamaIcon,
    OjamaWarningResult,
)


def _make_warning(p1_total: int, p2_total: int) -> tuple[
    OjamaWarningResult, OjamaWarningResult,
]:
    """テスト用に最小限の warning ペアを作る (icons は省略)。"""
    p1 = OjamaWarningResult(side=SIDE_P1, icons=(), total_count=p1_total)
    p2 = OjamaWarningResult(side=SIDE_P2, icons=(), total_count=p2_total)
    return p1, p2


def test_initial_state_zero() -> None:
    """初期状態では pending_ojama は両側 0。"""
    tracker = OjamaTimelineTracker()
    assert tracker.get_pending_ojama(SIDE_P1) == 0
    assert tracker.get_pending_ojama(SIDE_P2) == 0
    assert tracker.get_history() == []


def test_update_records_history_with_changes() -> None:
    """update() で履歴と change_from_previous が正しく記録される。"""
    tracker = OjamaTimelineTracker()
    tracker.update(0.0, _make_warning(0, 0))
    tracker.update(1.0, _make_warning(30, 0))   # P1 に 30 個飛んできた
    tracker.update(2.0, _make_warning(30, 60))  # P2 に 60 個追加
    tracker.update(3.0, _make_warning(0, 60))   # P1 が相殺し 30 → 0

    history = tracker.get_history()
    assert len(history) == 8  # 4 フレーム × 2 サイド

    # P1 の差分のみ取り出す
    p1_changes = tracker.filter_history(side=SIDE_P1, only_changes=True)
    assert [e.change_from_previous for e in p1_changes] == [30, -30]
    assert tracker.get_pending_ojama(SIDE_P1) == 0
    assert tracker.get_pending_ojama(SIDE_P2) == 60


def test_update_rejects_time_regression() -> None:
    """時刻が逆行する update は ValueError。"""
    tracker = OjamaTimelineTracker()
    tracker.update(5.0, _make_warning(0, 0))
    with pytest.raises(ValueError):
        tracker.update(4.0, _make_warning(0, 0))


def test_invalid_side_query() -> None:
    """未知の side を query すると ValueError。"""
    tracker = OjamaTimelineTracker()
    with pytest.raises(ValueError):
        tracker.get_pending_ojama("3P")


def test_reset_clears_state() -> None:
    """reset() で状態がクリアされる。"""
    tracker = OjamaTimelineTracker()
    tracker.update(0.0, _make_warning(30, 60))
    tracker.update(1.0, _make_warning(0, 60))
    tracker.reset()
    assert tracker.get_pending_ojama(SIDE_P1) == 0
    assert tracker.get_pending_ojama(SIDE_P2) == 0
    assert tracker.get_history() == []


def test_filter_history_by_side_and_changes() -> None:
    """filter_history は side と only_changes で絞り込める。"""
    tracker = OjamaTimelineTracker()
    tracker.update(0.0, _make_warning(0, 0))
    tracker.update(1.0, _make_warning(0, 0))    # 変化なし
    tracker.update(2.0, _make_warning(60, 0))   # P1 増加

    all_p1 = tracker.filter_history(side=SIDE_P1, only_changes=False)
    assert len(all_p1) == 3
    only_changed_p1 = tracker.filter_history(side=SIDE_P1, only_changes=True)
    assert len(only_changed_p1) == 1
    assert only_changed_p1[0].change_from_previous == 60
