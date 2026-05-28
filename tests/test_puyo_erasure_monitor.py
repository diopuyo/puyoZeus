"""PuyoErasureMonitor 単体テスト。

STABLE 中の「色→EMPTY」遷移が alert に記録され、
STABLE 以外はスキップされることを確認する。
"""
from __future__ import annotations

import pytest

from src.board import COLOR_BLUE, COLOR_EMPTY, COLOR_RED, COLOR_UNKNOWN, Board
from src.board_state_machine import BoardState
from src.puyo_erasure_monitor import PuyoErasureMonitor


def _board_with(row: int, col: int, color: int) -> Board:
    """指定 cell のみ color を設定した Board を返す。"""
    b = Board()
    b.set(row, col, color)
    return b


class TestPuyoErasureMonitor:
    """PuyoErasureMonitor テスト。"""

    def test_initial_count_is_zero(self) -> None:
        """初期状態で alert カウントは 0。"""
        mon = PuyoErasureMonitor()
        assert mon.count() == 0

    def test_stable_color_to_empty_fires_alert(self) -> None:
        """STABLE 中に色→EMPTY で alert が記録される。"""
        mon = PuyoErasureMonitor()
        prev = _board_with(5, 2, COLOR_RED)
        curr = _board_with(5, 2, COLOR_EMPTY)
        mon.update(100, BoardState.STABLE, prev, curr)
        assert mon.count() >= 1
        assert any(r == 5 and c == 2 for (fi, r, c) in mon.alerts)

    def test_non_stable_is_skipped(self) -> None:
        """STABLE 以外は alert を記録しない。"""
        mon = PuyoErasureMonitor()
        prev = _board_with(5, 2, COLOR_RED)
        curr = _board_with(5, 2, COLOR_EMPTY)
        for state in [
            BoardState.TSUMO_FALL,
            BoardState.CHAIN,
            BoardState.OJAMA_FALL,
            BoardState.EFFECT,
            BoardState.MENU,
        ]:
            mon.update(0, state, prev, curr)
        assert mon.count() == 0

    def test_none_boards_are_skipped(self) -> None:
        """prev or curr が None なら skip。"""
        mon = PuyoErasureMonitor()
        mon.update(0, BoardState.STABLE, None, _board_with(0, 0, COLOR_EMPTY))
        mon.update(0, BoardState.STABLE, _board_with(0, 0, COLOR_RED), None)
        assert mon.count() == 0

    def test_reset_clears_alerts(self) -> None:
        """reset() で alert がクリアされる。"""
        mon = PuyoErasureMonitor()
        prev = _board_with(1, 1, COLOR_BLUE)
        curr = _board_with(1, 1, COLOR_EMPTY)
        mon.update(10, BoardState.STABLE, prev, curr)
        assert mon.count() > 0
        mon.reset()
        assert mon.count() == 0

    def test_to_dict_contains_count(self) -> None:
        """to_dict に p_to_e_count が含まれる。"""
        mon = PuyoErasureMonitor()
        prev = _board_with(3, 3, COLOR_RED)
        curr = _board_with(3, 3, COLOR_EMPTY)
        mon.update(50, BoardState.STABLE, prev, curr)
        d = mon.to_dict()
        assert "p_to_e_count" in d
        assert d["p_to_e_count"] == mon.count()
        assert "alerts" in d

    def test_color_to_color_no_alert(self) -> None:
        """色→別の色 (EMPTY でない) は alert しない。"""
        mon = PuyoErasureMonitor()
        prev = _board_with(5, 2, COLOR_RED)
        curr = _board_with(5, 2, COLOR_BLUE)
        # T2 の上書きで実際には同色になるはずだが、monitor 自体は alert しない
        mon.update(100, BoardState.STABLE, prev, curr)
        assert mon.count() == 0

    def test_unknown_to_empty_no_alert(self) -> None:
        """UNKNOWN → EMPTY は物理上あり得るため alert しない。"""
        mon = PuyoErasureMonitor()
        prev = _board_with(0, 0, COLOR_UNKNOWN)
        curr = _board_with(0, 0, COLOR_EMPTY)
        mon.update(0, BoardState.STABLE, prev, curr)
        assert mon.count() == 0
