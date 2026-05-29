"""StableTransitionMonitor のテスト (Phase 1 / Task C2).

テスト方針:
    - stateless: 各テストは fresh instance で独立実行
    - 1P/2P 分離: monitor は side ごとに独立
    - 既存テスト 2114 件への副作用ゼロを確認 (import 変化なし)
"""
from __future__ import annotations

import pytest

from src.board import Board, BOARD_ROWS, BOARD_COLS, COLOR_RED, COLOR_BLUE, COLOR_EMPTY
from src.stable_transition_monitor import (
    STABLE_TRANSITION_DROP_THRESHOLD,
    StableTransitionMonitor,
    _count_puyo,
    _has_physics_event,
)


# ============================
# ユーティリティ
# ============================

def _make_board(puyo_count: int, color: int = COLOR_RED) -> Board:
    """指定個数のぷよを下から詰めた Board を返す。"""
    b = Board()
    filled = 0
    # 下 (row=12) から上 (row=0) に向かって埋める
    for row in range(BOARD_ROWS - 1, -1, -1):
        for col in range(BOARD_COLS):
            if filled >= puyo_count:
                break
            b.set(row, col, color)
            filled += 1
        if filled >= puyo_count:
            break
    return b


# ============================
# _count_puyo ユーティリティ
# ============================

class TestCountPuyo:
    def test_empty_board(self) -> None:
        """空盤面のぷよ数 = 0。"""
        b = Board()
        assert _count_puyo(b) == 0

    def test_filled_board(self) -> None:
        """10 個のぷよを持つ盤面のカウント。"""
        b = _make_board(10)
        assert _count_puyo(b) == 10


# ============================
# _has_physics_event ユーティリティ
# ============================

class TestHasPhysicsEvent:
    def test_no_events(self) -> None:
        """イベントなし → False。"""
        assert not _has_physics_event([], stable_start_t=5.0)

    def test_chain_event_in_window(self) -> None:
        """ウィンドウ内の chain_start → True。"""
        events = [(4.0, "chain_start")]
        assert _has_physics_event(events, stable_start_t=5.0)

    def test_chain_event_outside_window(self) -> None:
        """ウィンドウ外 (3 秒超前) のイベント → False。"""
        events = [(1.0, "chain_start")]
        assert not _has_physics_event(events, stable_start_t=5.0)

    def test_ojama_event_in_window(self) -> None:
        """ojama_land → True。"""
        events = [(4.5, "ojama_land")]
        assert _has_physics_event(events, stable_start_t=5.0)


# ============================
# StableTransitionMonitor 本体
# ============================

class TestStableTransitionMonitor:
    def test_first_stable_no_alert(self) -> None:
        """初回 STABLE 復帰は前 board なし → alert なし。"""
        mon = StableTransitionMonitor()
        board = _make_board(20)
        alerts = mon.on_stable_start(100, 1.0, board)
        assert alerts == []

    def test_no_alert_within_threshold(self) -> None:
        """THRESHOLD 以内の減少 (= 2 cell 以内) → alert なし。"""
        mon = StableTransitionMonitor()
        board_before = _make_board(20)
        board_after = _make_board(18)  # -2 = THRESHOLD 以内
        mon.on_stable_end(50, board_before)
        alerts = mon.on_stable_start(100, 1.0, board_after)
        assert alerts == []

    def test_alert_on_phantom_disappearance(self) -> None:
        """イベント無しで大幅減少 → alert が 1 件。"""
        mon = StableTransitionMonitor()
        board_before = _make_board(30)
        board_after = _make_board(10)  # -20 >> THRESHOLD
        mon.on_stable_end(50, board_before)
        alerts = mon.on_stable_start(100, 1.0, board_after)
        assert len(alerts) == 1
        # alert は TransitionDropAlert オブジェクト
        assert alerts[0].drop == 20
        assert alerts[0].prev_count == 30
        assert alerts[0].curr_count == 10

    def test_no_alert_on_normal_chain(self) -> None:
        """連鎖イベントあり大幅減少 → alert なし (= 連鎖消去の正常経路)。"""
        mon = StableTransitionMonitor()
        board_before = _make_board(30)
        board_after = _make_board(10)
        mon.on_stable_end(50, board_before)
        # NON-STABLE 中に chain_start イベント
        mon.on_non_stable_event("chain_start", 60, t_sec=0.9)
        alerts = mon.on_stable_start(100, 1.0, board_after)
        assert alerts == []

    def test_ojama_event_skips_alert(self) -> None:
        """ojama 着地イベントあり減少 → alert なし。"""
        mon = StableTransitionMonitor()
        board_before = _make_board(25)
        board_after = _make_board(5)
        mon.on_stable_end(50, board_before)
        mon.on_non_stable_event("ojama_land", 60, t_sec=0.85)
        alerts = mon.on_stable_start(100, 1.0, board_after)
        assert alerts == []

    def test_reset_clears_state(self) -> None:
        """reset() 後は前 board が消え、 次の on_stable_start は alert なし。"""
        mon = StableTransitionMonitor()
        board_before = _make_board(30)
        board_after = _make_board(5)
        mon.on_stable_end(50, board_before)
        mon.reset()
        # reset 後なので alert は出ない (= 初回 STABLE 扱い)
        alerts = mon.on_stable_start(100, 1.0, board_after)
        assert alerts == []
        # 累積 alert も消えている
        assert mon.get_all_alerts() == []

    def test_cumulative_alerts_accumulate(self) -> None:
        """複数回の phantom disappearance が累積される。"""
        mon = StableTransitionMonitor()
        for i in range(3):
            board_before = _make_board(30)
            board_after = _make_board(5)
            mon.on_stable_end(i * 100, board_before)
            mon.on_stable_start(i * 100 + 50, float(i), board_after)
        assert len(mon.get_all_alerts()) == 3

    def test_threshold_boundary_no_alert(self) -> None:
        """DROP = THRESHOLD + 0 → THRESHOLD 以内なので alert なし。"""
        # STABLE_TRANSITION_DROP_THRESHOLD = 2 → drop=2 は alert なし
        mon = StableTransitionMonitor()
        board_before = _make_board(10)
        board_after = _make_board(10 - STABLE_TRANSITION_DROP_THRESHOLD)
        mon.on_stable_end(0, board_before)
        alerts = mon.on_stable_start(10, 0.5, board_after)
        assert alerts == []

    def test_threshold_boundary_alert(self) -> None:
        """DROP = THRESHOLD + 1 → alert が出る。"""
        mon = StableTransitionMonitor()
        board_before = _make_board(10)
        board_after = _make_board(10 - STABLE_TRANSITION_DROP_THRESHOLD - 1)
        mon.on_stable_end(0, board_before)
        alerts = mon.on_stable_start(10, 0.5, board_after)
        assert len(alerts) == 1

    def test_to_dict_structure(self) -> None:
        """to_dict() が必要なキーを持つ。"""
        mon = StableTransitionMonitor()
        d = mon.to_dict()
        assert "alert_count" in d
        assert "alerts" in d
        assert "has_last_stable_board" in d
