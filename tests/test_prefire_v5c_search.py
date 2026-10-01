"""着弾表と逐次会計の全フィールド一致を検証する。"""
from dataclasses import replace

import numpy as np
import pytest

from src import prefire_v5_search as base
from src import prefire_v5c_search as fast
from src import prefire_v5b_search as exact
from tests.test_prefire_v5_search import position


@pytest.mark.parametrize('attacker', (0, 1))
@pytest.mark.parametrize('elapsed', (0., 95., 96., 120., 240.))
def test_table_matches_sequential_with_pending_and_margin(attacker: int, elapsed: float) -> None:
    table = fast.TransitionTable()
    for pending in (0, 1, 6, 29, 30, 31, 72):
        left = replace(position(), pending=pending, score=280, path=((1, 0),))
        right = replace(position(), pending=7, score=40, path=((2, 1),))
        assert table.resolve(left, right, attacker, elapsed) == base.resolve(left, right, attacker, elapsed)
        assert table.resolve(left, right, attacker, elapsed) == base.resolve(left, right, attacker, elapsed)


def test_partial_landing_depends_on_opponent_board() -> None:
    receiver = position()
    boards = []
    for color in range(1, 6):
        board = np.zeros((13, 6), dtype=np.int8)
        board[-1, 0] = color
        attacker = replace(position(), board=board.tobytes(), score=70)
        result = fast.TransitionTable().resolve(attacker, receiver, 0, 0.)
        assert result == base.resolve(attacker, receiver, 0, 0.)
        boards.append(result.sides[1].board)
    assert len(set(boards)) > 1


def test_full_rows_share_landing_but_keep_audit_path() -> None:
    table = fast.TransitionTable()
    left, right = replace(position(), score=420), position()
    first = table.resolve(left, right, 0, 0.)
    left = replace(left, board=bytes([1])+bytes(77), path=((3, 2),))
    second = table.resolve(left, right, 0, 0.)
    assert second == base.resolve(left, right, 0, 0.)
    assert first.sides[1].board == second.sides[1].board
    assert second.sides[0].path == left.path
    assert table.land.cache_info().hits > 0


def test_side_cache_reuses_unchanged_side() -> None:
    fast.side_options.cache_clear()
    for side in (0, 1):
        assert fast.side_options(position(), 1, side) is fast.side_options(position(), 1, side)
    assert fast.side_options.cache_info().hits == 2


@pytest.mark.parametrize('cell,expected', ((0, False), (1, True), (9, True), (10, False)))
def test_byte_death_matches_original(cell: int, expected: bool) -> None:
    board = np.zeros((13, 6), dtype=np.int8)
    board[1, 2] = cell
    assert fast.dead(board.tobytes()) is expected
    assert fast.dead(board.tobytes()) == base.sim._dead(board)


@pytest.mark.parametrize('attacker', (0, 1))
@pytest.mark.parametrize('pending', (0, 7))
def test_three_hand_choices_match_sequential(attacker: int, pending: int) -> None:
    states = (position(), replace(position(), pending=pending))
    def evaluator(exchange: base.Exchange, elapsed: float) -> float:
        return sum((i+1)*sum(p.board) for i, p in enumerate(exchange.sides)) / 1000
    sequential = exact.choose(states, attacker, 0., evaluator)
    table = fast.TransitionTable()
    actual = exact.choose(states, attacker, 0., evaluator,
                          resolver=table.resolve, options_fn=fast.side_options)
    assert actual == sequential
