"""D-v2h2bの完走上限と既存判定の互換性を検証する。"""
from __future__ import annotations

import argparse
import inspect
from types import SimpleNamespace as NS

import pytest

from src.exchange_post_counter_recent import stage_freshness
from src.exchange_prefire_stage_timeout import STAGE_INTERVAL_SEC
from tests.test_falsedeath_v2h2 import strict_case


@pytest.mark.parametrize('elapsed', [10., 100., 10000.])
def test_completed_chain_stays_fresh(elapsed: float) -> None:
    engine, projection, overlay, chain, context = strict_case()
    engine.cap_completed = True
    result = engine.prove(projection, overlay, 0, (), 171, 1, context, elapsed)
    check = result['completion_prefix_check']
    assert check['accepted'] and check['fresh'] and result['dead']
    assert check['estimated_current_stage'] == chain.predicted_chain_count
    assert check['stage_lag'] == 0


@pytest.mark.parametrize('lag,accepted', [(0, True), (1, True), (2, False), (3, False)])
def test_incomplete_chain_lag_is_unchanged(lag: int, accepted: bool) -> None:
    chain = NS(trigger_sec=0., predicted_chain_count=15)
    entry = dict(stages={7: {}})
    stamp = (6 + lag) * STAGE_INTERVAL_SEC + 0.01
    old = stage_freshness(chain, entry, stamp)
    new = stage_freshness(chain, entry, stamp, cap_completed=True)
    assert old == new
    assert new['fresh'] is accepted


@pytest.mark.parametrize('stamp', [None, -1., float('nan'), float('inf')])
def test_invalid_time_is_still_unobserved(stamp: float | None) -> None:
    chain = NS(trigger_sec=0., predicted_chain_count=9)
    assert not stage_freshness(chain, dict(stages={9: {}}), stamp, cap_completed=True)['fresh']


def test_missing_observations_still_block() -> None:
    engine, projection, overlay, _, context = strict_case()
    engine.cap_completed = True
    engine.step_observations.clear()
    assert not engine.prove(projection, overlay, 0, (), 171, 1, context, 100.)['dead']


def test_default_off_keeps_completed_expiry() -> None:
    chain = NS(trigger_sec=0., predicted_chain_count=9)
    assert not stage_freshness(chain, dict(stages={9: {}}), 100.)['fresh']


def test_flags_are_optional_and_mode_is_additive() -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay
    from scripts.replay_exchange_event_20260926 import replay
    from scripts.visualize_advantage_overlay import generate
    from scripts.falsedeath_replay_20261001 import enabled_flags
    for function in (ExchangeEventOverlay, replay, generate):
        assert inspect.signature(function).parameters['death_freshness_cap'].default is False
    assert enabled_flags('v2h2b') == enabled_flags('v2h2') + ['--death-freshness-cap']


def test_cap_requires_strict_integrity() -> None:
    from src.exchange_event_cli import validate_v2_options
    from src.exchange_event_overlay import ExchangeEventOverlay
    with pytest.raises(SystemExit):
        validate_v2_options(argparse.ArgumentParser(), NS(death_freshness_cap=True))
    overlay = ExchangeEventOverlay.__new__(ExchangeEventOverlay)
    overlay._landing_projection = NS(post_counter_bound=None)
    with pytest.raises(ValueError, match='D-v2h2'):
        overlay._initialize_prefix_integrity(False, cap_completed=True)


@pytest.mark.parametrize('predicted', [None, 0, -1])
def test_unknown_prediction_keeps_existing_freshness(predicted: int | None) -> None:
    chain = NS(trigger_sec=0., predicted_chain_count=predicted)
    entry = dict(stages={9: {}})
    assert stage_freshness(chain, entry, 100., cap_completed=True) == stage_freshness(chain, entry, 100.)
