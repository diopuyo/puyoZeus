"""E35⁗の一定差・段増分・欠測・既存E35‴との独立性を検証する。"""
from __future__ import annotations

import argparse
import inspect
from types import SimpleNamespace as NS
from typing import Any

import pytest

from src.exchange_post_counter_steps import compare_steps, observe_steps, record_formula
from tests.test_falsedeath_v2f import scenario, prove


def step_case() -> tuple[Any, ...]:
    """reviewの8〜10段目の実測値と、一定260点低い予測で死亡を判定する。"""
    case = scenario(False)
    engine, projection, _, chain, _ = case
    engine.completion_step_integrity = True
    prefix = {8: 21640, 9: 29320, 10: 40620}
    projection.safety.recovery.entries[1]['options'][0]['prefix'] = prefix
    chain.predicted_chain_count, chain.predicted_final_score = 10, 40620
    chain.formula_total, chain.display_score = 40880, 41880
    engine.step_observations[(0, 1)] = dict(stages={n: dict(formula=v+260, display=v+260)
        for n, v in prefix.items()}, conflicts=[], last_count=10)
    return case


def test_constant_offset_review_proves_death() -> None:
    case = step_case()
    result = prove(case)
    assert result['dead']
    check = case[0].audit[-1]['completion_prefix_check']
    assert check['reason'] == 'steps_consistent'
    assert check['candidates'][0]['comparisons'][0]['predicted_increment'] == 7680
    assert check['candidates'][0]['comparisons'][0]['offsets'] == dict(formula=260, display=260)


def test_mia7_stage_number_mismatch_does_not_prove() -> None:
    case = step_case()
    case[1].safety.recovery.entries[1]['options'][0]['prefix'] = (40,)
    case[3].predicted_chain_count, case[3].predicted_final_score = 1, 40
    assert not prove(case)['dead']
    assert case[0].audit[-1]['completion_prefix_check']['reason'] == 'step_mismatch'


@pytest.mark.parametrize('count', [8, 9, 10])
@pytest.mark.parametrize('channel', ['formula', 'display'])
def test_any_single_wrong_stage_rejects(count: int, channel: str) -> None:
    case = step_case()
    case[0].step_observations[(0, 1)]['stages'][count][channel] += 1
    assert not prove(case)['dead']


@pytest.mark.parametrize('missing', ['all', 'single', 'gap', 'scores', 'candidate'])
def test_unobserved_rejects(missing: str) -> None:
    case = step_case()
    stages = case[0].step_observations[(0, 1)]['stages']
    if missing == 'all':
        stages.clear()
    elif missing == 'single':
        del stages[8], stages[9]
    elif missing == 'gap':
        del stages[9]
    elif missing == 'scores':
        for row in stages.values():
            row.update(formula=None, display=None)
    else:
        case[1].safety.recovery.entries.clear()
    assert not prove(case)['dead']


@pytest.mark.parametrize('channel', ['formula', 'display'])
def test_either_observed_channel_can_verify(channel: str) -> None:
    other = 'display' if channel == 'formula' else 'formula'
    stages = {2: {channel: 500, other: None}, 3: {channel: 1140, other: None}}
    assert compare_steps({2: 240, 3: 880}, stages)['accepted']


def test_later_matching_increment_does_not_erase_earlier_mismatch() -> None:
    stages = {n: dict(formula=v, display=None) for n, v in enumerate((300, 801, 1441), 1)}
    result = compare_steps({1: 40, 2: 540, 3: 1180}, stages)
    assert not result['accepted'] and result['comparisons'][-1]['observed_increments']['formula'] == 640


def test_actual_review_has_earlier_wrong_third_step_and_missing_sixth() -> None:
    """後半の一定差だけでは、前半で発生した260点の不一致を解消しない。"""
    prefix = {1: 100, 2: 420, 3: 1060, 4: 2340, 5: 4900, 7: 15240, 8: 21640, 9: 29320}
    totals = {1: 100, 2: 420, 3: 1320, 4: 2600, 5: 5160, 7: 15500, 8: 21900, 9: 29580}
    result = compare_steps(prefix, {n: dict(formula=v, display=v) for n, v in totals.items()})
    assert not result['accepted'] and result['mismatch'] and result['missing']
    assert result['comparisons'][1]['predicted_increment'] == 640
    assert result['comparisons'][1]['observed_increments'] == dict(formula=900, display=900)
    assert result['comparisons'][-1]['predicted_increment'] == 7680


def test_formula_revision_remains_conflict() -> None:
    entry = dict(stages={}, conflicts=[], last_count=None)
    record_formula(entry, 1, 300)
    record_formula(entry, 1, 301)
    record_formula(entry, 1, 300)
    assert entry['conflicts'] == ['formula_revision']


def test_estimated_formula_cannot_supply_stage_evidence() -> None:
    case = step_case()
    event = NS(mechanism='formula', trigger_sec=0., chain_count=1, total_score=40)
    entries = {}
    observe_steps(case[2], NS(p1=NS(chain_event=event), p2=NS(chain_event=None)), entries)
    assert not entries[(0, 1)]['stages']


def test_display_lag_is_assigned_to_accepted_formula_stage() -> None:
    case = step_case()
    entries = {(0, 1): dict(stages={8: dict(formula=21900, display=None)}, conflicts=[], last_count=8)}
    case[3].formula_total, case[3].display_score = 21900, 22900
    event = NS(mechanism='formula_read', trigger_sec=0., chain_count=9, total_score=29580)
    observe_steps(case[2], NS(p1=NS(chain_event=event), p2=NS(chain_event=None)), entries)
    assert entries[(0, 1)]['stages'][8]['display'] == 21900
    assert entries[(0, 1)]['stages'][9]['display'] is None


def test_v2f_still_rejects_constant_offset() -> None:
    case = step_case()
    case[0].completion_step_integrity = False
    case[0].completion_prefix_integrity = True
    case[0].formula_observations[(0, 1)] = (10, 40880)
    assert not prove(case)['dead']
    assert case[0].audit[-1]['completion_prefix_check']['reason'] == 'prefix_mismatch'


def test_default_off_and_fixed_configuration() -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay
    from scripts.replay_exchange_event_20260926 import replay
    from scripts.visualize_advantage_overlay import generate
    from scripts.falsedeath_replay_20261001 import enabled_flags
    for function in (ExchangeEventOverlay, replay, generate):
        assert inspect.signature(function).parameters['post_counter_completion_step_integrity'].default is False
    assert enabled_flags('v2g') == enabled_flags('v2e') + ['--post-counter-completion-step-integrity']


@pytest.mark.parametrize('missing', ['post_counter_death_bound', 'post_counter_current_puyo_integrity_v2c'])
def test_missing_dependency_rejected(missing: str) -> None:
    from src.exchange_event_cli import validate_v2_options
    options = dict(post_counter_death_bound=True, post_counter_current_puyo_integrity_v2c=True,
                   post_counter_completion_step_integrity=True)
    options[missing] = False
    with pytest.raises(SystemExit):
        validate_v2_options(argparse.ArgumentParser(), NS(**options))


def test_v2f_and_v2g_cannot_be_combined() -> None:
    from src.exchange_event_cli import validate_v2_options
    with pytest.raises(SystemExit):
        validate_v2_options(argparse.ArgumentParser(), NS(post_counter_death_bound=True,
            post_counter_current_puyo_integrity_v2c=True, post_counter_completion_step_integrity=True,
            post_counter_completion_prefix_integrity=True))
