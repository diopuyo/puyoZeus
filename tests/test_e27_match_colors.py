"""試合4色の取得に孤立誤読や未来フレームを使わない。"""
from __future__ import annotations

from types import SimpleNamespace as NS
import pytest
from src.board import Board
from src.board_state_machine import BoardState as S
from src.match_color_evidence import MatchColorEvidence


@pytest.mark.parametrize('counts,expected', [({}, ()), ({1:5, 2:5, 3:5}, ()),
    ({1:5, 2:5, 3:5, 4:5}, (1, 2, 3, 4)),
    ({1:50, 2:50, 3:50, 4:50, 5:1}, (1, 2, 3, 4)),
    ({1:50, 2:50, 3:50, 4:1, 5:50}, (1, 2, 3, 5)),
    ({1:50, 2:50, 3:50, 4:1, 5:1}, ())])
def test_ranking_and_ambiguity(counts: dict, expected: tuple) -> None:
    tracker = MatchColorEvidence()
    tracker.counts.update(counts)
    assert tracker.active() == expected


def test_only_current_stable_cells_are_observed() -> None:
    tracker, board = MatchColorEvidence(), Board()
    board._grid[-1] = [1, 2, 3, 4, 9, 10]
    side = NS(state=S.CHAIN, confirmed_board=board)
    tracker.observe(side)
    assert tracker.active() == ()
    side.state = S.STABLE
    tracker.observe(side)
    assert tracker.active() == (1, 2, 3, 4)
    assert set(tracker.counts) == {1, 2, 3, 4}
