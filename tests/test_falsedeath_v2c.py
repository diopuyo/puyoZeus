"""D-v2cの観測限定分岐・証拠保存・誤指定拒否・正の対照を検証する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from src.board import Board
from src.exchange_post_counter_bound import PostCounterDeathBound


@pytest.mark.parametrize('missing', [None, 'state', 'placed', 'next', 'motion', 'ended', 'signal'])
def test_omit_requires_existing_observations(missing: str | None) -> None:
    engine = PostCounterDeathBound(current_puyo_integrity_v2c=True)
    engine.states = ('CHAIN' if missing != 'state' else 'GRAVITY_SETTLE', None)
    engine.next_motion = (missing == 'motion', False)
    queue = (1, 2, 3, 4)
    engine.queues = (queue, queue)
    observed = dict(firing_placed=missing != 'placed', next_unchanged=missing != 'next',
                    ended=missing == 'ended')
    signal = NS(current_puyo_observation=lambda q: observed)
    overlay = NS(_signals={} if missing == 'signal' else {1: signal})
    chain = NS(chain_id=1, trigger_sec=1., end_signal_sec=None)
    actual, audit = engine.current_queue(overlay, chain, 0)
    assert audit['unknown_current_pair_omitted_self_chain'] == (missing is None)
    assert len(actual) == (4 if missing is None else 6)
    assert audit['current_puyo_observation']['state'] == engine.states[0]


def test_real_end_sensor_exposes_existing_baseline() -> None:
    from scripts.visualize_advantage_overlay import _ExchangeEventEndSignals
    signal = object.__new__(_ExchangeEventEndSignals)
    signal.side = 0
    signal.sensor = NS(_abs_next_at_start=[(1, 2)], _queue_successors=[{(3, 4)}],
        _abs_baseline_ready=[True], _abs_saw_chain=[True], _abs_ended=[False],
        _abs_end_kind=[None], _abs_pending_frames=[0])
    assert signal.current_puyo_observation((1, 2, 3, 4))['next_unchanged']
    assert not signal.current_puyo_observation((3, 4, 1, 2))['next_unchanged']
    signal.sensor._abs_baseline_ready[0] = False
    assert not signal.current_puyo_observation((1, 2, 3, 4))['firing_placed']


@pytest.mark.parametrize('option', ['score_attribution_v2', 'post_counter_input_integrity_v2',
    'post_counter_current_puyo_integrity', 'post_counter_current_puyo_integrity_v2c'])
def test_e35_off_rejects_v2_flags(option: str) -> None:
    from src.exchange_event_cli import validate_v2_options
    from src.exchange_event_overlay import ExchangeEventOverlay
    with pytest.raises(SystemExit):
        validate_v2_options(argparse.ArgumentParser(), NS(**{option: True}))
    with pytest.raises(ValueError, match='post-counter-death-bound'):
        ExchangeEventOverlay(None, None, None, **{option: True})


def test_independent_evidence_saved(tmp_path: Path) -> None:
    from tests.test_falsedeath_v2 import score_tracker
    tracker, chain = score_tracker()
    tracker.finalize('1P', 3., 1000.)
    path = tmp_path/'events.jsonl'
    tracker.save(path)
    saved = json.loads(path.read_text())['chains'][0]
    assert saved['independent_score_required'] is True
    assert saved['finalized_evidence_score'] == 1000.


@pytest.mark.parametrize('v2c', [False, True])
def test_positive_death_remains_dead_with_unknown(v2c: bool) -> None:
    board = Board()
    board._grid[-8:] = 9
    chain = NS(chain_id=1, trigger_sec=0., end_signal_sec=None,
               predicted_final_score=0., predicted_final_board=None)
    overlay = NS(_signals={}, _game=0,
        tracker=NS(latest_chain=lambda side: chain, _score_elapsed=0.))
    engine = PostCounterDeathBound(current_puyo_integrity=not v2c, current_puyo_integrity_v2c=v2c)
    engine.queues = ((1, 2, 3, 4),)*2
    engine.colors = [NS(active=lambda: (1, 2, 3, 4))]*2
    projection = NS(_known_budget=lambda *a: True, _chaining=lambda *a: False)
    context = dict(hidden=[None]*2, certain=[True]*2, credit=[0]*2,
                   verified=[True]*2, replies=[board]*2)
    result = engine.prove(projection, overlay, 0, (), 171, 1, context, 1.)
    assert result['dead'] and result['unknown_current_pair_prepended']
    assert engine.audit[-1]['unknown_current_pair_prepended']


def test_v2c_fixed_configuration() -> None:
    from scripts.falsedeath_replay_20261001 import enabled_flags, input_record
    assert enabled_flags('v2c') == ['--death-input-integrity', '--score-attribution-v2',
                                  '--post-counter-current-puyo-integrity-v2c']
    assert input_record('v2c', 'review') == input_record('all', 'review')


def test_omissions_only_match_same_game_side_and_future() -> None:
    from scripts.falsedeath_v2c_omissions import classify, counts
    row = dict(game=1, side='2P', chain_id=2, t_sec=10., dead=True, cached=True,
        unknown_current_pair_omitted_self_chain=True, current_puyo_observation={})
    terminals = [dict(game=1, side='2P', stamp=9.), dict(game=2, side='2P', stamp=11.),
                 dict(game=1, side='1P', stamp=11.)]
    result = classify([row], terminals, 11., 20.)
    assert counts(result)['not_observed'] == 1 and not result[0]['in_display']
    terminals.append(dict(game=1, side='2P', stamp=12.))
    assert counts(classify([row], terminals, 0., 20.))['observed'] == 1


def test_old_current_flag_always_prepends() -> None:
    engine = PostCounterDeathBound(current_puyo_integrity=True)
    engine.states = ('CHAIN', 'CHAIN')
    engine.queues = ((1, 2, 3, 4),)*2
    queue, evidence = engine.current_queue(NS(), NS(), 0)
    assert queue == (0, 0, 1, 2, 3, 4)
    assert evidence['unknown_current_pair_prepended']


def test_observation_revision_tracks_baseline_and_end() -> None:
    engine = PostCounterDeathBound(current_puyo_integrity_v2c=True)
    chain = NS(chain_id=1, trigger_sec=0., end_signal_sec=None)
    observed = dict(firing_placed=False, next_unchanged=False, ended=False)
    overlay = NS(_signals={1: NS(current_puyo_observation=lambda q: observed)},
                 tracker=NS(latest_chain=lambda side: chain))
    engine.refresh_current_observation(overlay)
    revision = engine.revision
    engine.refresh_current_observation(overlay)
    assert engine.revision == revision
    observed.update(firing_placed=True, next_unchanged=True)
    engine.refresh_current_observation(overlay)
    assert engine.revision == revision+1
    chain.end_signal_sec = 1.
    engine.refresh_current_observation(overlay)
    assert engine.revision == revision+2
