"""Phase 5 の状態伝播と候補契約。"""
from dataclasses import replace

import numpy as np
import pytest

from src import prefire_v5_search as v5
from src.board import Board


def position(queue: tuple[int, ...] = (1, 2, 3, 4, 1, 3)) -> v5.Position:
    """空盤面の既知状態。"""
    return v5.Position(Board()._grid.astype(np.int8).tobytes(), queue)


@pytest.mark.parametrize('queue', [(), (0, 0), (1, 0), (9, 2), (1, 2), (1, 2, 0, 0)])
def test_missing(queue: tuple[int, ...]) -> None:
    state = position(queue)
    assert v5.status(state.board, queue) == v5.MISSING
    assert not v5.candidates(state)


def test_firing_transition() -> None:
    board = Board()
    board._grid[-1, :4] = 1
    state = replace(position(), board=board._grid.astype(np.int8).tobytes())
    assert v5.status(state.board, state.queue) == v5.FIRING
    assert not v5.candidates(state)


def test_wait_consumes_and_places() -> None:
    states = v5.candidates(position(), depth=1, k=None)
    assert len(states) == 22
    assert all(s.consumed == 1 and s.queue == (3, 4, 1, 3) for s in states)
    assert all(np.count_nonzero(np.frombuffer(s.board, np.int8)) == 2 for s in states)


def test_response_removes_puyos() -> None:
    board = Board()
    board._grid[-1, :3] = 1
    start = replace(position((1, 2)), board=board._grid.astype(np.int8).tobytes())
    response = next(p for p in v5.extend(start) if p.score)
    result = v5.resolve(position(), response, 0, 0.0)
    assert result.sides[1].consumed == 1
    assert result.sides[1].queue == ()
    assert np.count_nonzero(np.frombuffer(result.sides[1].board, np.int8) == 1) == 0


@pytest.mark.parametrize('pending', [0, 1, 29, 30, 31, 60, 120])
def test_pending_not_lost(pending: int) -> None:
    state = replace(position(), pending=pending)
    result = v5.resolve(state, position(), 0, 0.0)
    assert result.sides[0].pending + result.dropped[0] == pending


def test_both_pending_offset() -> None:
    left = replace(position(), pending=10, score=1400)
    right = replace(position(), pending=5, score=700)
    result = v5.resolve(left, right, 0, 0.0)
    assert result.cancelled == (10, 10)
    assert result.dropped == (0, 5)


def test_fixed_k() -> None:
    assert len(v5.candidates(position(), depth=1)) == v5.TOP_K
    assert v5.candidates(position(), depth=1) == v5.candidates(position(), depth=1)
