"""H1の途中開始・開始観測・次試合復帰と観測死亡の独立性を検証する。"""
from __future__ import annotations

import inspect
from types import SimpleNamespace as NS

import pytest

from src.board import Board
from src.board_state_machine import BoardState
from src.exchange_history_integrity import MatchHistoryIntegrity

MIDGAME_SCORE = 1000

def sides(left: int, right: int) -> tuple:
    """下段から指定個数だけある確定盤面を作る。"""
    result = []
    for count in (left, right):
        board = Board()
        if count:
            board._grid.flat[-count:] = 1
        result.append(NS(state=BoardState.STABLE, confirmed_board=board, score=0))
    return tuple(result)


@pytest.mark.parametrize('proof', ['single_death_proof', 'multi_landing', 'post_counter_bound'])
@pytest.mark.parametrize('complete', [False, True])
def test_all_computed_proofs_require_start(proof: str, complete: bool) -> None:
    guard = MatchHistoryIntegrity()
    guard.observe(sides(0, 0) if complete else sides(20, 18), 0, 10.)
    guard.observe(sides(2, 0), 0, 11.)
    value = dict(source='unavoidable_death', p1=.98, dead_sides=['2P'], t_sec=20.,
                 **{proof: [dict(dead=True)]})
    audit = [dict(dead=True)]
    guard.apply(value, .72, audit)
    assert (value['source'] == 'unavoidable_death') == complete
    assert value['p1'] == (.98 if complete else .72)
    assert value['history_integrity']['history_complete'] == complete
    assert audit[0]['history_integrity']['blocked'] != complete


def test_boundary_recovers_but_midgame_clear_does_not() -> None:
    guard = MatchHistoryIntegrity()
    guard.observe(sides(10, 20), 0, 1.)
    guard.observe(sides(0, 0), 0, 2.)
    guard.observe(sides(2, 2), 0, 3.)
    assert not guard.evidence()['history_complete']
    guard.observe(sides(0, 0), 1, 4.)
    guard.observe(sides(2, 0), 1, 5.)
    assert guard.evidence() == dict(history_complete=True, match_start_observed_sec=5.)
    assert set(guard.games) == {1}


def test_missing_empty_or_large_first_board_is_incomplete() -> None:
    for sequence in ((sides(2, 0),), (sides(0, 0), sides(4, 0))):
        guard = MatchHistoryIntegrity()
        for stamp, pair in enumerate(sequence):
            guard.observe(pair, 0, float(stamp))
        assert not guard.evidence()['history_complete']


def test_boundary_zero_score_residual_is_not_first_placement() -> None:
    guard = MatchHistoryIntegrity()
    guard.observe(sides(20, 10), 0, 1.)
    residual = sides(6, 71)
    for side in residual:
        side.score = 0
    guard.observe(residual, 1, 2.)
    assert not guard.evidence()['history_complete']
    guard.observe(sides(0, 0), 1, 3.)
    guard.observe(sides(2, 0), 1, 4.)
    assert guard.evidence()['history_complete']
    assert guard.games[1]['boundary_residual_frames'] == 1


@pytest.mark.parametrize('score', [None, MIDGAME_SCORE])
def test_initial_all_clear_without_zero_score_is_incomplete(score: int | None) -> None:
    guard = MatchHistoryIntegrity()
    clear = sides(0, 0)
    for side in clear:
        side.score = score
        side.state = BoardState.CHAIN
    guard.observe(clear, 0, 1.)
    for side in clear:
        side.state = BoardState.STABLE
    guard.observe(clear, 0, 2.)
    guard.observe(sides(2, 0), 0, 3.)
    assert not guard.evidence()['history_complete']
    assert guard.evidence()['match_start_observed_sec'] is None


def test_observed_death_unchanged() -> None:
    guard = MatchHistoryIntegrity()
    guard.observe(sides(20, 10), 0, 1.)
    value = dict(source='confirmed_death', p1=1., dead_sides=['2P'], t_sec=2.)
    guard.apply(value, .5, [])
    assert value['source'] == 'confirmed_death' and value['p1'] == 1.
    assert not value['history_integrity']['blocked']


def test_default_off_and_fixed_configuration() -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay
    from scripts.replay_exchange_event_20260926 import replay
    from scripts.visualize_advantage_overlay import generate
    from scripts.falsedeath_replay_20261001 import enabled_flags, input_record
    for function in (ExchangeEventOverlay, replay, generate):
        assert inspect.signature(function).parameters['death_history_integrity'].default is False
    assert enabled_flags('v2d') == enabled_flags('v2c') + ['--death-history-integrity']
    assert input_record('v2d', 'review') == input_record('v2c', 'review')
