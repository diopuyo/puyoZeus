"""E35⁗′の直近2段・過去段監査・欠測と既定OFFを検証する。"""
from __future__ import annotations

import argparse
import inspect
from types import SimpleNamespace as NS
from typing import Any

import pytest

from src.exchange_post_counter_recent import compare_recent
from tests.test_falsedeath_v2f import prove
from tests.test_falsedeath_v2g import step_case


def recent_case() -> tuple[Any, ...]:
    """過去2→3段に260点差があり、7〜9段の累積が実読されたreview型。"""
    case = step_case()
    engine, projection, _, chain, _ = case
    engine.completion_step_integrity, engine.completion_recent_integrity = False, True
    prefix = {1: 100, 2: 420, 3: 1060, 4: 2340, 5: 4900, 7: 15240, 8: 21640, 9: 29320}
    totals = {1: 100, 2: 420, 3: 1320, 4: 2600, 5: 5160, 7: 15500, 8: 21900, 9: 29580}
    projection.safety.recovery.entries[1]['options'][0]['prefix'] = prefix
    chain.predicted_chain_count, chain.predicted_final_score = 9, 29320
    chain.formula_total, chain.display_score = 29580, 30580
    engine.step_observations[(0, 1)] = dict(stages={n: dict(formula=v, display=v)
        for n, v in totals.items()}, conflicts=[], last_count=9)
    return case


def test_review_earlier_mismatch_recent_two_match_proves() -> None:
    case = recent_case()
    assert prove(case)['dead']
    check = case[0].audit[-1]['completion_prefix_check']
    assert check['accepted'] and check['earlier_mismatch']
    assert check['candidates'][0]['recent_counts'] == [7, 8, 9]
    assert [r['predicted_increment'] for r in check['candidates'][0]['comparisons']] == [6400, 7680]


def test_mia7_missing_formula_does_not_prove() -> None:
    case = recent_case()
    case[0].step_observations.clear()
    assert not prove(case)['dead']
    assert case[0].audit[-1]['completion_prefix_check']['reason'] == 'recent_unobserved'


@pytest.mark.parametrize('stage', [8, 9])
@pytest.mark.parametrize('channel', ['formula', 'display'])
def test_either_recent_increment_wrong_does_not_prove(stage: int, channel: str) -> None:
    case = recent_case()
    stages = case[0].step_observations[(0, 1)]['stages']
    for count in range(stage, 10):
        stages[count][channel] += 1
    assert not prove(case)['dead']


def test_stage_numbers_different_does_not_prove() -> None:
    case = recent_case()
    case[1].safety.recovery.entries[1]['options'][0]['prefix'] = (40,)
    case[3].predicted_chain_count, case[3].predicted_final_score = 1, 40
    assert not prove(case)['dead']


@pytest.mark.parametrize('missing', [7, 8, 9])
def test_latest_two_increments_must_be_observed(missing: int) -> None:
    case = recent_case()
    del case[0].step_observations[(0, 1)]['stages'][missing]
    assert not prove(case)['dead']


def test_one_increment_is_insufficient() -> None:
    stages = {8: dict(formula=21900, display=None), 9: dict(formula=29580, display=None)}
    assert not compare_recent({8: 21640, 9: 29320}, stages)['accepted']


@pytest.mark.parametrize('channel', ['formula', 'display'])
def test_either_channel_can_supply_two_increments(channel: str) -> None:
    other = 'display' if channel == 'formula' else 'formula'
    stages = {n: {channel: n*100+260, other: None} for n in (7, 8, 9)}
    assert compare_recent({n: n*100 for n in stages}, stages)['accepted']


def test_v2g_still_rejects_past_mismatch() -> None:
    case = recent_case()
    case[0].completion_recent_integrity, case[0].completion_step_integrity = False, True
    assert not prove(case)['dead']


def test_default_off_and_fixed_configuration() -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay
    from scripts.replay_exchange_event_20260926 import replay
    from scripts.visualize_advantage_overlay import generate
    from scripts.falsedeath_replay_20261001 import enabled_flags
    for function in (ExchangeEventOverlay, replay, generate):
        assert inspect.signature(function).parameters['post_counter_completion_recent_integrity'].default is False
    assert enabled_flags('v2h') == enabled_flags('v2e') + ['--post-counter-completion-recent-integrity']


@pytest.mark.parametrize('other', ['prefix', 'step'])
def test_cannot_combine_old_integrity(other: str) -> None:
    from src.exchange_event_cli import validate_v2_options
    options = dict(post_counter_death_bound=True, post_counter_current_puyo_integrity_v2c=True,
                   post_counter_completion_recent_integrity=True)
    options[f'post_counter_completion_{other}_integrity'] = True
    with pytest.raises(SystemExit):
        validate_v2_options(argparse.ArgumentParser(), NS(**options))


@pytest.mark.parametrize('missing', ['post_counter_death_bound', 'post_counter_current_puyo_integrity_v2c'])
def test_dependencies_required(missing: str) -> None:
    from src.exchange_event_cli import validate_v2_options
    options = dict(post_counter_death_bound=True, post_counter_current_puyo_integrity_v2c=True,
                   post_counter_completion_recent_integrity=True)
    options[missing] = False
    with pytest.raises(SystemExit):
        validate_v2_options(argparse.ArgumentParser(), NS(**options))
