"""H1′の境界証拠・誤読拒否・既定OFFの互換性を検証する。"""
from __future__ import annotations

import inspect

import pytest

from src.board_state_machine import BoardState
from src.exchange_history_integrity import MatchHistoryIntegrity
from tests.test_falsedeath_v2d import sides


def observe(guard: MatchHistoryIntegrity, game: int, stamp: float,
            scores: tuple = (0, 0)) -> None:
    """開始直後にMENUが続く実際の認識状態を模す。"""
    pair = sides(20, 20)
    for side, score in zip(pair, scores):
        side.state = BoardState.MENU
        side.score = score
    guard.observe(pair, game, stamp)


def test_formal_boundary_zero_scores_complete_with_evidence() -> None:
    guard = MatchHistoryIntegrity(boundary_score_start=True)
    observe(guard, 0, 1., (1000, 2000))
    observe(guard, 1, 2.)
    assert guard.evidence()['history_complete']
    evidence = guard.evidence()['boundary_score_evidence']
    assert evidence == dict(boundary_sec=2., previous_game=0,
        previous_nonzero_scores=[dict(score=1000, t_sec=1.), dict(score=2000, t_sec=1.)],
        zero_observed_secs=[2., 2.], accepted=True)
    value = dict(source='unavoidable_death', p1=1., dead_sides=['2P'], t_sec=3.)
    proofs = [dict(dead=True)]
    guard.apply(value, .5, proofs)
    assert value['history_integrity']['boundary_score_evidence'] == evidence
    assert guard.audit[-1]['boundary_score_evidence'] == evidence
    assert proofs[0]['history_integrity']['boundary_score_evidence'] == evidence


@pytest.mark.parametrize('previous', [(0, 0), (None, None), (1000, 0), (0, 2000)])
def test_zero_without_previous_nonzero_is_incomplete(previous: tuple) -> None:
    guard = MatchHistoryIntegrity(boundary_score_start=True)
    observe(guard, 0, 1., previous)
    observe(guard, 1, 2.)
    assert not guard.evidence()['history_complete']


def test_first_record_and_midgame_zero_are_incomplete() -> None:
    guard = MatchHistoryIntegrity(boundary_score_start=True)
    observe(guard, 10, 1.)
    observe(guard, 10, 2., (1000, 2000))
    observe(guard, 10, 3.)
    assert not guard.evidence()['history_complete']
    observe(guard, 12, 4.)
    assert not guard.evidence()['history_complete']


def test_late_zero_after_positive_start_is_not_start() -> None:
    guard = MatchHistoryIntegrity(boundary_score_start=True)
    observe(guard, 0, 1., (1000, 2000))
    observe(guard, 1, 2., (1, 0))
    observe(guard, 1, 3.)
    assert not guard.evidence()['history_complete']


def test_missing_score_waits_without_future_evidence() -> None:
    guard = MatchHistoryIntegrity(boundary_score_start=True)
    observe(guard, 0, 1., (1000, 2000))
    observe(guard, 1, 2., (0, None))
    old = guard.evidence()
    assert not old['history_complete']
    observe(guard, 1, 3.)
    assert guard.evidence()['history_complete']
    assert old['boundary_score_evidence']['zero_observed_secs'] == [2., None]
    assert guard.evidence()['match_start_observed_sec'] == 3.


def test_original_h1_is_preserved() -> None:
    guard = MatchHistoryIntegrity(boundary_score_start=True)
    guard.observe(sides(0, 0), 0, 1.)
    guard.observe(sides(2, 0), 0, 2.)
    assert guard.evidence()['history_complete']
    assert not guard.evidence()['boundary_score_evidence']['accepted']


def test_default_off_keeps_exact_evidence() -> None:
    guards = [MatchHistoryIntegrity(), MatchHistoryIntegrity(boundary_score_start=False)]
    for guard in guards:
        observe(guard, 0, 1., (1000, 2000))
        observe(guard, 1, 2.)
    assert guards[0].summary() == guards[1].summary()
    assert guards[0].evidence() == dict(history_complete=False, match_start_observed_sec=None)


def test_flag_defaults_and_fixed_configuration() -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay
    from scripts.replay_exchange_event_20260926 import replay
    from scripts.visualize_advantage_overlay import generate
    from scripts.falsedeath_replay_20261001 import enabled_flags, input_record
    for function in (ExchangeEventOverlay, replay, generate):
        assert inspect.signature(function).parameters['death_history_boundary_start'].default is False
    assert enabled_flags('v2e') == enabled_flags('v2d') + ['--death-history-boundary-start']
    assert input_record('v2e', 'review') == input_record('v2d', 'review')
